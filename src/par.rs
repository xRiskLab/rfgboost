//! Parallelism abstraction.
//!
//! Native targets use rayon. On emscripten/wasm the runtime (Pyodide) is
//! single-threaded, so rayon would try to spawn worker threads and panic at
//! the first parallel call. There we fall back to sequential iteration that
//! presents the same call-site API (`into_par_iter`, `par_iter`, `scope`,
//! `ThreadPoolBuilder`), so the rest of the crate is unchanged.

#[cfg(not(target_os = "emscripten"))]
pub use rayon::prelude::*;
#[cfg(not(target_os = "emscripten"))]
pub use rayon::{scope, ThreadPoolBuilder};

#[cfg(target_os = "emscripten")]
pub use seq::{scope, IntoParallelIterator, IntoParallelRefIterator, ThreadPoolBuilder};

#[cfg(target_os = "emscripten")]
mod seq {
    /// Sequential stand-in for rayon's `into_par_iter()`.
    pub trait IntoParallelIterator {
        type Item;
        type Iter: Iterator<Item = Self::Item>;
        fn into_par_iter(self) -> Self::Iter;
    }

    impl<I: IntoIterator> IntoParallelIterator for I {
        type Item = I::Item;
        type Iter = I::IntoIter;
        fn into_par_iter(self) -> Self::Iter {
            self.into_iter()
        }
    }

    /// Sequential stand-in for rayon's `par_iter()` on slices and vectors.
    pub trait IntoParallelRefIterator<'a> {
        type Item: 'a;
        type Iter: Iterator<Item = &'a Self::Item>;
        fn par_iter(&'a self) -> Self::Iter;
    }

    impl<'a, T: 'a> IntoParallelRefIterator<'a> for [T] {
        type Item = T;
        type Iter = std::slice::Iter<'a, T>;
        fn par_iter(&'a self) -> Self::Iter {
            self.iter()
        }
    }

    impl<'a, T: 'a> IntoParallelRefIterator<'a> for Vec<T> {
        type Item = T;
        type Iter = std::slice::Iter<'a, T>;
        fn par_iter(&'a self) -> Self::Iter {
            self.iter()
        }
    }

    /// Sequential stand-in for `rayon::Scope`: `spawn` runs the body inline.
    pub struct Scope;

    impl Scope {
        pub fn spawn<F>(&self, f: F)
        where
            F: FnOnce(&Scope),
        {
            f(self);
        }
    }

    /// Sequential stand-in for `rayon::scope`.
    pub fn scope<F, R>(f: F) -> R
    where
        F: FnOnce(&Scope) -> R,
    {
        f(&Scope)
    }

    /// No-op stand-in for `rayon::ThreadPoolBuilder` (single-threaded wasm).
    #[derive(Default)]
    pub struct ThreadPoolBuilder;

    #[derive(Debug)]
    pub struct ThreadPoolBuildError;

    impl ThreadPoolBuilder {
        pub fn new() -> Self {
            ThreadPoolBuilder
        }
        pub fn num_threads(self, _n: usize) -> Self {
            self
        }
        pub fn build_global(self) -> Result<(), ThreadPoolBuildError> {
            Ok(())
        }
    }
}
