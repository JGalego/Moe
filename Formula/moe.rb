# A Homebrew formula, for a tap rather than core.
#
#     brew tap jgalego/moe https://github.com/JGalego/Moe
#     brew install moe
#
# Built from source: the engine's fastest kernels depend on the host's SIMD, and a
# bottle would have to be built for the lowest common denominator.
class Moe < Formula
  desc "CPU inference for sparse mixture-of-experts language models"
  homepage "https://github.com/JGalego/Moe"
  url "https://github.com/JGalego/Moe/archive/refs/tags/v0.2.0.tar.gz"
  license "MIT"
  head "https://github.com/JGalego/Moe.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args
  end

  test do
    # The version, and a command that exercises real work without a download:
    # tokenize needs only a vocabulary.
    assert_match "moe", shell_output("#{bin}/moe --version")
    (testpath/"tokenizer.json").write <<~JSON
      {"model": {"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}}
    JSON
    assert_equal "2", shell_output("#{bin}/moe tokenize #{testpath}/tokenizer.json -p ab").strip
  end
end
