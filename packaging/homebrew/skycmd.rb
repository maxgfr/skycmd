# Homebrew formula template for the tap maxgfr/homebrew-tap (Formula/skycmd.rb).
#
# The release workflow fills the @VERSION@ and @SHA256_<target>@ placeholders from the release's
# SHA256SUMS (packaging/homebrew/render.sh) and pushes the result to the tap when the
# HOMEBREW_TAP_TOKEN secret is set. Install with:
#
#   brew install maxgfr/tap/skycmd
class Skycmd < Formula
  desc "Tiny System One decision models (GGUF): CLI and /v1/systemone HTTP server"
  homepage "https://maxgfr.github.io/skycmd/"
  version "@VERSION@"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/maxgfr/skycmd/releases/download/v#{version}/skycmd-#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "@SHA256_aarch64-apple-darwin@"
    end
    on_intel do
      url "https://github.com/maxgfr/skycmd/releases/download/v#{version}/skycmd-#{version}-x86_64-apple-darwin.tar.gz"
      sha256 "@SHA256_x86_64-apple-darwin@"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/maxgfr/skycmd/releases/download/v#{version}/skycmd-#{version}-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA256_aarch64-unknown-linux-gnu@"
    end
    on_intel do
      url "https://github.com/maxgfr/skycmd/releases/download/v#{version}/skycmd-#{version}-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA256_x86_64-unknown-linux-gnu@"
    end
  end

  resource "s1-pico" do
    url "https://github.com/maxgfr/skycmd/releases/download/v@VERSION@/s1-pico.gguf"
    sha256 "@SHA256_s1-pico.gguf@"
  end

  def install
    bin.install "skycmd"
    # a small model to try the CLI right away: $(brew --prefix)/share/skycmd/s1-pico.gguf
    resource("s1-pico").stage { (pkgshare/"models").install "s1-pico.gguf" }
  end

  def caveats
    <<~EOS
      A small model is installed at:
        #{opt_pkgshare}/models/s1-pico.gguf
      Start a server with:
        skycmd serve --model #{opt_pkgshare}/models/s1-pico.gguf --port 8080
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/skycmd --version")
    request = '{"state":"alt 12 vy -3","questions":{"go":{"type":"noul","instructions":"Should the drone climb?"}}}'
    out = pipe_output("#{bin}/skycmd decide --model #{pkgshare}/models/s1-pico.gguf -", request)
    assert_match "\"noul\"", out
  end
end
