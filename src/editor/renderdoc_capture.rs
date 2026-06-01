//! F5-in-editor RenderDoc capture. The dll is preloaded in `main::load_renderdoc` (before
//! the wgpu device exists) so the graphics hook installs; here we grab the in-application
//! API handle and trigger a single-frame capture on F5. The `.rdc` lands in the CWD next
//! to the exe — open it in the RenderDoc UI afterwards. No external launcher needed.
//!
//! The API handle holds a raw pointer (`!Send`/`!Sync`), so it lives as a `NonSend`
//! resource and its driver runs as a `NonSendMut` system on the main thread.

use bevy::prelude::*;
use renderdoc::{OverlayBits, RenderDoc, V141};

/// The in-application RenderDoc API handle. Absent (the system early-returns) when the dll
/// wasn't preloaded — e.g. RenderDoc isn't installed, or `fast`/dynamic_linking blocked the
/// hook. `V141` is RenderDoc 1.4.1, a baseline every modern install satisfies.
struct RenderDocApi(RenderDoc<V141>);

pub struct RenderDocCapturePlugin;

impl Plugin for RenderDocCapturePlugin {
    fn build(&self, app: &mut App) {
        // `RenderDoc::new()` succeeds only if the dll is already resident (preloaded in
        // main). On failure we simply don't insert the handle — F5 becomes a no-op and a
        // one-line hint is logged, rather than failing the build or the run.
        match RenderDoc::<V141>::new() {
            Ok(mut rd) => {
                // Kill the on-screen overlay (the FPS/capture-list text RenderDoc draws over
                // the swapchain): mask ALL bits off (and = empty) and OR nothing back in.
                // Captures still work; only the HUD is gone.
                rd.mask_overlay_bits(OverlayBits::empty(), OverlayBits::empty());

                // Write captures next to the project (rdoc/) instead of the system temp
                // dir. RenderDoc appends `_frameNNNNN.rdc` to this template.
                rd.set_capture_file_path_template("rdoc/adventure");

                app.insert_non_send_resource(RenderDocApi(rd));
                app.add_systems(Update, trigger_on_f5);
                info!("RenderDoc capture: API ready (overlay off) — F5 captures to rdoc/.");
            }
            Err(e) => {
                info!("RenderDoc capture: API unavailable ({e}); F5 capture disabled.");
            }
        }
    }
}

/// F5 → capture the next presented frame. RenderDoc names the file from the exe + frame
/// number and writes it to the CWD; the log line points at where to look.
fn trigger_on_f5(keyboard: Res<ButtonInput<KeyCode>>, mut rd: NonSendMut<RenderDocApi>) {
    if keyboard.just_pressed(KeyCode::F5) {
        rd.0.trigger_capture();
        info!("RenderDoc capture: triggered — .rdc will be written to rdoc/.");
    }
}
