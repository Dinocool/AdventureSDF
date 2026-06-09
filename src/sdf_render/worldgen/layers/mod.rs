//! Concrete generation layers. The Phase-1 slice ships one: [`height`], the CPU-authoritative base
//! terrain height. Future layers (climate, biome classification, caves, scatter) join here, each a
//! self-contained `super::layer::Layer` impl registered into the recipe palette.

pub mod erosion;
pub mod height;
