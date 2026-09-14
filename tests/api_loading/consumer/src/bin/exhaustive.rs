use cera::ModelHandle;

fn classify(handle: &ModelHandle) -> &'static str {
    match handle {
        ModelHandle::Generative(_) => "generative",
    }
}

fn main() {
    let _ = classify;
}
