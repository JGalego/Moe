//! Routing analysis: what a sparse model actually did with its experts.
//!
//! A trace is a flat list of `(position, layer, chosen experts)`. Everything
//! interesting is a statistic over that list — how much of the model a prompt
//! reached, how evenly it spread, which experts a domain never touches — so this
//! module folds counting, the summary table and the SVG into the engine rather
//! than leaving them to a script the binary cannot run.
//!
//! Colour follows the job: one hue with monotone lightness for magnitude, two
//! hues either side of a neutral grey for polarity, both interpolated in OKLab
//! so the steps are perceptually even rather than eyeballed.

use crate::model::Trace;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

const SURFACE: &str = "#0d1117";
const INK: &str = "#d6dde8";
const INK_DIM: &str = "#8892a2";
const GRID: &str = "#1c2430";
const AMBER: &str = "#e0973a";
const TEAL: &str = "#17b3a3";
/// Diverging midpoint. Achromatic by construction, so zero never reads as a hue.
const NEUTRAL: &str = "#323232";

// ------------------------------------------------------------------- counting

/// Selections per `(layer, expert)`, plus what the trace says about the model.
#[derive(Default, Clone)]
pub struct Counts {
    pub model: String,
    pub experts: usize,
    pub top_k: usize,
    /// Distinct token positions the trace covers.
    pub tokens: usize,
    counts: BTreeMap<(u32, u32), u64>,
}

impl Counts {
    /// Count an in-memory trace, as `moe run --trace` would have written it.
    pub fn from_trace(tr: &Trace, model: &str, experts: usize, top_k: usize) -> Counts {
        let mut c = Counts { model: model.into(), experts, top_k, tokens: 0, counts: BTreeMap::new() };
        let mut positions: Vec<u32> = Vec::new();
        for r in &tr.routes {
            positions.push(r.pos);
            for (e, _) in &r.experts {
                *c.counts.entry((r.layer, *e)).or_default() += 1;
            }
        }
        positions.sort_unstable();
        positions.dedup();
        c.tokens = positions.len();
        c
    }

    /// Count a JSONL trace file, header line included.
    pub fn read(path: &Path) -> std::io::Result<Counts> {
        use std::io::{BufRead, BufReader};
        let mut c = Counts::default();
        let mut positions: Vec<u32> = Vec::new();
        for line in BufReader::new(std::fs::File::open(path)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
            // The self-describing header line carries no position.
            if v.get("pos").is_none() {
                c.model = v["model"].as_str().unwrap_or("").to_string();
                c.experts = v["experts"].as_u64().unwrap_or(0) as usize;
                c.top_k = v["top_k"].as_u64().unwrap_or(0) as usize;
                continue;
            }
            let (pos, layer) = (v["pos"].as_u64().unwrap_or(0) as u32, v["layer"].as_u64().unwrap_or(0) as u32);
            positions.push(pos);
            for pair in v["experts"].as_array().into_iter().flatten() {
                if let Some(e) = pair[0].as_u64() {
                    *c.counts.entry((layer, e as u32)).or_default() += 1;
                }
            }
        }
        positions.sort_unstable();
        positions.dedup();
        c.tokens = positions.len();
        if c.experts == 0 {
            c.experts = c.counts.keys().map(|(_, e)| *e + 1).max().unwrap_or(0) as usize;
        }
        Ok(c)
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Routed layers, ascending. A dense layer never appears in a trace.
    pub fn layers(&self) -> Vec<u32> {
        let mut l: Vec<u32> = self.counts.keys().map(|(l, _)| *l).collect();
        l.dedup();
        l
    }

    pub fn count(&self, layer: u32, expert: u32) -> u64 {
        self.counts.get(&(layer, expert)).copied().unwrap_or(0)
    }

    /// Share of tokens that sent work to this expert, in `0..=1`.
    pub fn share(&self, layer: u32, expert: u32) -> f32 {
        self.count(layer, expert) as f32 / self.tokens.max(1) as f32
    }

    fn layer_counts(&self, layer: u32) -> Vec<u64> {
        (0..self.experts as u32).map(|e| self.count(layer, e)).collect()
    }

    /// Routing entropy for one layer, normalised so 1.0 is a perfectly even
    /// spread and 0.0 is every token choosing the same expert. This is the
    /// number that says whether a checkpoint's sparsity is real: an entropy
    /// near 1 means the router is using its whole capacity.
    pub fn entropy(&self, layer: u32) -> f32 {
        let c = self.layer_counts(layer);
        let total: u64 = c.iter().sum();
        if total == 0 || self.experts < 2 {
            return 0.0;
        }
        let h: f32 = c
            .iter()
            .filter(|n| **n > 0)
            .map(|n| {
                let p = *n as f32 / total as f32;
                -p * p.ln()
            })
            .sum();
        h / (self.experts as f32).ln()
    }

    /// Peak load over the load a uniform router would give each expert. 1.0 is
    /// balanced; 4.0 means the busiest expert did four times its share.
    pub fn peak_ratio(&self, layer: u32) -> f32 {
        let c = self.layer_counts(layer);
        let total: u64 = c.iter().sum();
        let peak = c.iter().copied().max().unwrap_or(0);
        if total == 0 {
            return 0.0;
        }
        peak as f32 * self.experts as f32 / total as f32
    }

    /// Experts this layer never selected.
    pub fn dead(&self, layer: u32) -> usize {
        self.layer_counts(layer).iter().filter(|n| **n == 0).count()
    }

    /// Experts never selected in any layer, as a fraction of the whole model.
    /// The fraction of a checkpoint a prompt never reads is the reason `pack
    /// --keep-experts` can make it smaller.
    pub fn coverage(&self) -> f32 {
        let layers = self.layers().len();
        if layers == 0 || self.experts == 0 {
            return 0.0;
        }
        self.counts.values().filter(|n| **n > 0).count() as f32 / (layers * self.experts) as f32
    }

    /// Every `(layer, expert)` this trace touched, busiest first.
    ///
    /// Routing is skewed, so the head of this list is where a residency budget
    /// buys the most: pinning it keeps the experts a workload actually uses out
    /// of the way of eviction.
    pub fn hottest(&self) -> Vec<(u32, u32)> {
        let mut v: Vec<(&(u32, u32), &u64)> = self.counts.iter().filter(|(_, n)| **n > 0).collect();
        v.sort_unstable_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        v.into_iter().map(|(k, _)| *k).collect()
    }

    /// A pruning plan keeping the `width` busiest experts in each layer.
    ///
    /// Every layer keeps the same *number*, because a config declares one expert
    /// count, but the sets differ — which is where the specialisation lives. A
    /// layer the trace barely exercised is topped up with its lowest-numbered
    /// unused experts, so the shape stays uniform rather than the model becoming
    /// unloadable.
    pub fn prune_plan(&self, layers: usize, width: usize, top_k: usize) -> crate::store::Prune {
        let width = width.clamp(top_k.max(1), self.experts.max(1));
        let keep = (0..layers)
            .map(|l| {
                let l = l as u32;
                let mut chosen: Vec<u32> = self.top(l, width).into_iter().map(|(e, _)| e).collect();
                // Top up with unused experts so every layer is the same width.
                for e in 0..self.experts as u32 {
                    if chosen.len() >= width {
                        break;
                    }
                    if !chosen.contains(&e) {
                        chosen.push(e);
                    }
                }
                chosen.sort_unstable();
                chosen
            })
            .collect();
        crate::store::Prune { keep, top_k }
    }

    /// Strongest `n` experts in a layer, share-descending.
    pub fn top(&self, layer: u32, n: usize) -> Vec<(u32, f32)> {
        let mut v: Vec<(u32, f32)> =
            (0..self.experts as u32).map(|e| (e, self.share(layer, e))).filter(|(_, s)| *s > 0.0).collect();
        v.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(n);
        v
    }

    /// The same numbers as the picture, so a trace is readable over ssh.
    pub fn report(&self, top: usize) -> String {
        let layers = self.layers();
        let mut s = String::new();
        let _ = writeln!(
            s,
            "{}{} tokens, {} routed layers, {} experts, top-{}",
            if self.model.is_empty() { String::new() } else { format!("{}: ", self.model) },
            self.tokens,
            layers.len(),
            self.experts,
            self.top_k
        );
        let _ = writeln!(
            s,
            "coverage {:.0}% of (layer, expert) pairs touched   mean entropy {:.2}   mean peak/uniform {:.1}x",
            100.0 * self.coverage(),
            layers.iter().map(|l| self.entropy(*l)).sum::<f32>() / layers.len().max(1) as f32,
            layers.iter().map(|l| self.peak_ratio(*l)).sum::<f32>() / layers.len().max(1) as f32,
        );
        let _ = writeln!(s, "\n  layer  entropy  peak  dead  busiest experts");
        for l in &layers {
            let busiest: Vec<String> =
                self.top(*l, top).iter().map(|(e, v)| format!("e{e}:{:.0}%", 100.0 * v)).collect();
            let _ = writeln!(
                s,
                "  {:<6} {:>7.2} {:>5.1}x {:>5}  {}",
                format!("L{l}"),
                self.entropy(*l),
                self.peak_ratio(*l),
                self.dead(*l),
                busiest.join("  ")
            );
        }
        s
    }
}

/// Signed difference between two runs' shares, for the diff view.
pub struct Diff {
    pub a: Counts,
    pub b: Counts,
}

impl Diff {
    pub fn layers(&self) -> Vec<u32> {
        let mut l = self.a.layers();
        l.extend(self.b.layers());
        l.sort_unstable();
        l.dedup();
        l
    }

    pub fn experts(&self) -> usize {
        self.a.experts.max(self.b.experts)
    }

    pub fn share(&self, layer: u32, expert: u32) -> f32 {
        self.a.share(layer, expert) - self.b.share(layer, expert)
    }

    /// Largest absolute divergence, which sets the colour scale.
    pub fn span(&self) -> f32 {
        self.layers()
            .iter()
            .flat_map(|l| (0..self.experts() as u32).map(move |e| (*l, e)))
            .map(|(l, e)| self.share(l, e).abs())
            .fold(0.0f32, f32::max)
            .max(1e-6)
    }

    /// Experts each side favours most, as `(layer, expert, delta)`.
    pub fn extremes(&self, n: usize) -> Vec<(u32, u32, f32)> {
        let mut v: Vec<(u32, u32, f32)> = self
            .layers()
            .iter()
            .flat_map(|l| (0..self.experts() as u32).map(move |e| (*l, e)))
            .map(|(l, e)| (l, e, self.share(l, e)))
            .filter(|(_, _, d)| d.abs() > 1e-6)
            .collect();
        v.sort_unstable_by(|x, y| y.2.abs().total_cmp(&x.2.abs()));
        v.truncate(n);
        v
    }

    pub fn report(&self, top: usize) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "difference over {} routed layers x {} experts (first {} tokens, second {} tokens)",
            self.layers().len(),
            self.experts(),
            self.a.tokens,
            self.b.tokens
        );
        let _ = writeln!(s, "widest divergence {:+.1}%\n", 100.0 * self.span());
        let _ = writeln!(s, "  layer  expert  delta   favours");
        for (l, e, d) in self.extremes(top) {
            let _ = writeln!(
                s,
                "  {:<6} {:<7} {:>+6.1}%  {}",
                format!("L{l}"),
                format!("e{e}"),
                100.0 * d,
                if d > 0.0 { "first" } else { "second" }
            );
        }
        s
    }
}

// --------------------------------------------------------------------- colour

type Lab = [f32; 3];

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn hex_to_oklab(h: &str) -> Lab {
    let byte = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).unwrap_or(0) as f32 / 255.0;
    let (r, g, b) = (srgb_to_linear(byte(1)), srgb_to_linear(byte(3)), srgb_to_linear(byte(5)));
    let l = (0.412_221_5 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
    let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

fn oklab_to_hex(lab: Lab) -> String {
    let [big_l, a, b] = lab;
    let l = (big_l + 0.396_337_78 * a + 0.215_803_76 * b).powi(3);
    let m = (big_l - 0.105_561_346 * a - 0.063_854_17 * b).powi(3);
    let s = (big_l - 0.089_484_18 * a - 1.291_485_5 * b).powi(3);
    let rgb = [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ];
    let mut out = String::from("#");
    for c in rgb {
        let v = (255.0 * linear_to_srgb(c.clamp(0.0, 1.0)).clamp(0.0, 1.0)).round() as u8;
        let _ = write!(out, "{v:02x}");
    }
    out
}

fn lerp(a: Lab, b: Lab, t: f32) -> Lab {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/// A sequential ramp from near the surface up to `to`. On a dark surface "near
/// zero" is the dark end, so lightness rises monotonically with magnitude.
fn ramp(to: &str, steps: usize) -> Vec<String> {
    let (lo, hi) = (hex_to_oklab(SURFACE), hex_to_oklab(to));
    let floor = 0.04;
    (0..steps)
        .map(|i| {
            let t = i as f32 / (steps.max(2) - 1) as f32;
            // Ease the low end so the first non-zero cell is already visible.
            oklab_to_hex(lerp(lo, hi, floor + (1.0 - floor) * t))
        })
        .collect()
}

/// Two hues either side of a neutral grey, equal steps per arm. Each arm runs
/// outward from the midpoint, not from the surface: starting at the surface
/// would make the steps nearest zero darker than zero itself.
fn diverging(steps: usize) -> Vec<String> {
    let half = steps / 2;
    let mid = hex_to_oklab(NEUTRAL);
    let arm = |to: &str| -> Vec<String> {
        let hi = hex_to_oklab(to);
        (0..=half).map(|i| oklab_to_hex(lerp(mid, hi, i as f32 / half.max(1) as f32))).collect()
    };
    let (cool, warm) = (arm(TEAL), arm(AMBER));
    let mut out: Vec<String> = cool.into_iter().rev().collect();
    out.pop();
    out.push(NEUTRAL.to_string());
    out.extend(warm.into_iter().skip(1));
    out
}

/// The "nothing" step must be the least chromatic, or zero reads as a value.
fn midpoint_is_neutral(c: &[String]) -> bool {
    let chroma: Vec<f32> = c
        .iter()
        .map(|h| {
            let lab = hex_to_oklab(h);
            lab[1] * lab[1] + lab[2] * lab[2]
        })
        .collect();
    let lowest = (0..chroma.len()).min_by(|a, b| chroma[*a].total_cmp(&chroma[*b])).unwrap_or(0);
    lowest == c.len() / 2
}

fn lightness_is_monotone(c: &[String]) -> bool {
    c.windows(2).all(|w| hex_to_oklab(&w[1])[0] >= hex_to_oklab(&w[0])[0] - 1e-6)
}

// --------------------------------------------------------------------- render

struct Grid<'a> {
    title: String,
    subtitle: String,
    layers: Vec<u32>,
    experts: usize,
    colours: Vec<String>,
    vmin: f32,
    vmax: f32,
    legend: (String, String),
    at: &'a dyn Fn(u32, u32) -> f32,
}

fn render(g: &Grid) -> String {
    let (cell, gap) = (13.0f32, 2.0f32);
    let (pad_l, pad_t, pad_r, pad_b) = (62.0f32, 62.0f32, 20.0f32, 64.0f32);
    let w = pad_l + g.experts as f32 * cell + pad_r;
    let h = pad_t + g.layers.len() as f32 * cell + pad_b;
    let mut o = String::new();
    let _ = write!(
        o,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\" \
         font-family=\"'SF Mono',Menlo,Consolas,monospace\">\n\
         <rect width=\"{w}\" height=\"{h}\" fill=\"{SURFACE}\"/>\n\
         <text x=\"{pad_l}\" y=\"26\" fill=\"{INK}\" font-size=\"13\" font-weight=\"600\">{}</text>\n\
         <text x=\"{pad_l}\" y=\"44\" fill=\"{INK_DIM}\" font-size=\"11\">{}</text>\n",
        escape(&g.title),
        escape(&g.subtitle),
    );

    let colour_of = |v: f32| -> &str {
        if g.vmax <= g.vmin {
            return &g.colours[g.colours.len() / 2];
        }
        let t = (v - g.vmin) / (g.vmax - g.vmin);
        let i = (t * (g.colours.len() - 1) as f32 + 0.5).clamp(0.0, (g.colours.len() - 1) as f32) as usize;
        &g.colours[i]
    };

    for (row, layer) in g.layers.iter().enumerate() {
        let y = pad_t + row as f32 * cell;
        if layer % 4 == 0 {
            let _ = writeln!(
                o,
                "<text x=\"{}\" y=\"{}\" fill=\"{INK_DIM}\" font-size=\"9.5\" text-anchor=\"end\">L{layer}</text>",
                pad_l - 8.0,
                y + cell - 4.0
            );
        }
        for e in 0..g.experts as u32 {
            let v = (g.at)(*layer, e);
            // A 2px surface gap keeps adjacent cells from reading as one block.
            let _ = writeln!(
                o,
                "<rect x=\"{}\" y=\"{y}\" width=\"{}\" height=\"{}\" rx=\"2\" fill=\"{}\">\
                 <title>layer {layer}, expert {e}: {:+.1}%</title></rect>",
                pad_l + e as f32 * cell,
                cell - gap,
                cell - gap,
                colour_of(v),
                100.0 * v
            );
        }
    }
    for e in (0..g.experts).step_by(8) {
        let _ = writeln!(
            o,
            "<text x=\"{}\" y=\"{}\" fill=\"{INK_DIM}\" font-size=\"9.5\" text-anchor=\"middle\">{e}</text>",
            pad_l + e as f32 * cell + (cell - gap) / 2.0,
            pad_t - 8.0
        );
    }
    let _ = writeln!(
        o,
        "<text x=\"{}\" y=\"{}\" fill=\"{INK_DIM}\" font-size=\"10\" text-anchor=\"middle\">expert</text>",
        pad_l + g.experts as f32 * cell / 2.0,
        h - 34.0
    );

    // A sequential or diverging ramp is unreadable without a scale.
    let (lw, lx, ly) = (168.0f32, pad_l, h - 22.0);
    let step = lw / g.colours.len() as f32;
    for (i, c) in g.colours.iter().enumerate() {
        let _ = writeln!(
            o,
            "<rect x=\"{:.2}\" y=\"{ly}\" width=\"{:.2}\" height=\"9\" fill=\"{c}\"/>",
            lx + i as f32 * step,
            step + 0.6
        );
    }
    let _ = write!(
        o,
        "<text x=\"{}\" y=\"{}\" fill=\"{INK_DIM}\" font-size=\"9.5\" text-anchor=\"end\">{}</text>\n\
         <text x=\"{}\" y=\"{}\" fill=\"{INK_DIM}\" font-size=\"9.5\">{}</text>\n\
         <rect x=\"{lx}\" y=\"{ly}\" width=\"{lw}\" height=\"9\" fill=\"none\" stroke=\"{GRID}\"/>\n</svg>\n",
        lx - 6.0,
        ly + 8.0,
        escape(&g.legend.0),
        lx + lw + 6.0,
        ly + 8.0,
        escape(&g.legend.1),
    );
    o
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// One run as a heatmap: share of tokens selecting each expert in each layer.
pub fn heatmap(c: &Counts, title: Option<&str>) -> String {
    let colours = ramp(AMBER, 12);
    debug_assert!(lightness_is_monotone(&colours), "sequential ramp is not monotone in lightness");
    let layers = c.layers();
    let vmax = layers
        .iter()
        .flat_map(|l| (0..c.experts as u32).map(move |e| (*l, e)))
        .map(|(l, e)| c.share(l, e))
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let at = |l: u32, e: u32| c.share(l, e);
    render(&Grid {
        title: title.map(String::from).unwrap_or_else(|| {
            if c.model.is_empty() {
                "expert routing".into()
            } else {
                format!("expert routing — {}", c.model)
            }
        }),
        subtitle: format!(
            "{} tokens x {} layers, top-{} of {}; share of tokens selecting each expert",
            c.tokens,
            layers.len(),
            c.top_k,
            c.experts
        ),
        layers,
        experts: c.experts,
        colours,
        vmin: 0.0,
        vmax,
        legend: ("0%".into(), format!("{:.0}%", 100.0 * vmax)),
        at: &at,
    })
}

/// Two runs as their difference: amber favours the first, teal the second.
pub fn diffmap(d: &Diff, labels: (&str, &str), title: Option<&str>) -> String {
    let colours = diverging(15);
    let mid = colours.len() / 2;
    debug_assert!(midpoint_is_neutral(&colours), "diverging midpoint is not the least chromatic step");
    debug_assert!(lightness_is_monotone(&colours[..=mid].iter().rev().cloned().collect::<Vec<_>>()), "cool arm dips");
    debug_assert!(lightness_is_monotone(&colours[mid..]), "warm arm dips");
    let span = d.span();
    let at = |l: u32, e: u32| d.share(l, e);
    let model = if d.a.model.is_empty() { String::new() } else { format!(" — {}", d.a.model) };
    render(&Grid {
        title: title.map(String::from).unwrap_or_else(|| format!("expert routing — difference{model}")),
        subtitle: format!(
            "{} ({} tokens) vs {} ({} tokens); amber favours the first, teal the second",
            labels.0, d.a.tokens, labels.1, d.b.tokens
        ),
        layers: d.layers(),
        experts: d.experts(),
        colours,
        vmin: -span,
        vmax: span,
        legend: (format!("-{:.0}%", 100.0 * span), format!("+{:.0}%", 100.0 * span)),
        at: &at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Route;

    fn trace(routes: &[(u32, u32, &[u32])]) -> Trace {
        Trace {
            tokens: Vec::new(),
            routes: routes
                .iter()
                .map(|(pos, layer, experts)| Route {
                    pos: *pos,
                    layer: *layer,
                    experts: experts.iter().map(|e| (*e, 0.5)).collect(),
                })
                .collect(),
        }
    }

    #[test]
    fn counts_shares_and_layers() {
        let c = Counts::from_trace(&trace(&[(0, 0, &[1, 2]), (1, 0, &[1, 3]), (0, 1, &[0, 1])]), "m", 4, 2);
        assert_eq!(c.tokens, 2);
        assert_eq!(c.layers(), vec![0, 1]);
        assert_eq!(c.count(0, 1), 2);
        assert!((c.share(0, 1) - 1.0).abs() < 1e-6);
        assert!((c.share(0, 2) - 0.5).abs() < 1e-6);
        assert_eq!(c.count(0, 0), 0);
    }

    /// A uniform layer must score entropy 1 and peak 1; a degenerate one, 0 and n.
    #[test]
    fn entropy_brackets_are_tight() {
        let even = Counts::from_trace(&trace(&[(0, 0, &[0, 1]), (1, 0, &[2, 3])]), "m", 4, 2);
        assert!((even.entropy(0) - 1.0).abs() < 1e-5, "{}", even.entropy(0));
        assert!((even.peak_ratio(0) - 1.0).abs() < 1e-5);
        assert_eq!(even.dead(0), 0);

        let one = Counts::from_trace(&trace(&[(0, 0, &[2]), (1, 0, &[2])]), "m", 4, 1);
        assert!(one.entropy(0) < 1e-6);
        assert!((one.peak_ratio(0) - 4.0).abs() < 1e-5);
        assert_eq!(one.dead(0), 3);
    }

    #[test]
    fn coverage_counts_touched_pairs() {
        // 2 layers x 4 experts = 8 pairs; 3 distinct pairs touched.
        let c = Counts::from_trace(&trace(&[(0, 0, &[0, 1]), (0, 1, &[0])]), "m", 4, 2);
        assert!((c.coverage() - 3.0 / 8.0).abs() < 1e-6);
    }

    #[test]
    fn diff_is_signed_and_finds_extremes() {
        let a = Counts::from_trace(&trace(&[(0, 0, &[0])]), "m", 4, 1);
        let b = Counts::from_trace(&trace(&[(0, 0, &[1])]), "m", 4, 1);
        let d = Diff { a, b };
        assert!((d.share(0, 0) - 1.0).abs() < 1e-6);
        assert!((d.share(0, 1) + 1.0).abs() < 1e-6);
        assert!((d.span() - 1.0).abs() < 1e-6);
        assert_eq!(d.extremes(2).len(), 2);
    }

    /// The colour contract the picture rests on, checked rather than eyeballed.
    #[test]
    fn ramps_are_perceptually_sound() {
        assert!(lightness_is_monotone(&ramp(AMBER, 12)));
        let d = diverging(15);
        let mid = d.len() / 2;
        assert!(midpoint_is_neutral(&d));
        assert!(lightness_is_monotone(&d[..=mid].iter().rev().cloned().collect::<Vec<_>>()));
        assert!(lightness_is_monotone(&d[mid..]));
    }

    #[test]
    fn oklab_round_trips_hex() {
        for h in [SURFACE, AMBER, TEAL, "#ffffff", "#000000"] {
            let back = oklab_to_hex(hex_to_oklab(h));
            assert_eq!(back, h, "{h} -> {back}");
        }
    }

    #[test]
    fn heatmap_is_wellformed_svg() {
        let c = Counts::from_trace(&trace(&[(0, 0, &[0, 1]), (1, 0, &[1, 2])]), "test", 4, 2);
        let s = heatmap(&c, None);
        assert!(s.starts_with("<svg"));
        assert!(s.trim_end().ends_with("</svg>"));
        // One cell per (layer, expert), plus the legend swatches.
        assert_eq!(s.matches("<title>").count(), 4);
    }
}
