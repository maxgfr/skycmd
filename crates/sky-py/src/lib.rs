//! Python bindings of the skycmd decision engine.
//!
//! ```python
//! import skycmd
//! m = skycmd.Model("s1-pico.gguf")
//! r = m.decide({"state": "alt 12", "questions": {"go": {"type": "noul", "instructions": "Climb?"}}})
//! r["answers"]["go"]["noul"]                   # probability of "yes"
//! m.decide_raw("choice", "Which action?", ["flap", "wait"], "alt 12")  # [p_flap, p_wait]
//! ```

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyString};
use sky_infer::{Engine, QType};

fn to_py_err(e: sky_infer::Error) -> PyErr {
    match e {
        sky_infer::Error::InvalidRequest(_)
        | sky_infer::Error::Format(_)
        | sky_infer::Error::Unsupported(_)
        | sky_infer::Error::Missing(_)
        | sky_infer::Error::Gguf(_) => PyValueError::new_err(e.to_string()),
        sky_infer::Error::Io(e) => PyOSError::new_err(e.to_string()),
    }
}

/// A loaded decision model (GGUF: s1 `modern-bert` or s2 `skycmd-s2`).
///
/// `Model(source)` takes a file path (`str` or `os.PathLike`) or the file content (`bytes`,
/// `bytearray`). Calls are thread-safe; they release the GIL while the model runs.
#[pyclass(module = "skycmd", name = "Model", frozen)]
struct Model {
    engine: Mutex<Engine>,
    path: Option<String>,
}

impl Model {
    fn engine(&self) -> MutexGuard<'_, Engine> {
        // a panic inside the engine poisons the lock; the engine itself has no broken invariant
        self.engine.lock().unwrap_or_else(|p| p.into_inner())
    }
}

#[pymethods]
impl Model {
    #[new]
    fn new(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (bytes, path) = if let Ok(b) = source.cast::<PyBytes>() {
            (b.as_bytes().to_vec(), None)
        } else if let Ok(b) = source.cast::<PyByteArray>() {
            (b.to_vec(), None)
        } else {
            let p: PathBuf = source.extract().map_err(|_| {
                PyValueError::new_err("Model() expects a path (str, os.PathLike) or bytes")
            })?;
            let bytes = std::fs::read(&p)
                .map_err(|e| PyOSError::new_err(format!("cannot read {}: {e}", p.display())))?;
            (bytes, Some(p.display().to_string()))
        };
        let engine = py.detach(|| Engine::from_gguf(&bytes)).map_err(to_py_err)?;
        Ok(Model {
            engine: Mutex::new(engine),
            path,
        })
    }

    /// Answer a System One request (the `POST /v1/systemone` body) and return the response.
    ///
    /// `request` is a dict (`{"state": ..., "questions": {...}}`) or its JSON text. The result is
    /// a dict `{"model", "answers", "usage"}`.
    fn decide<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let json = py.import("json")?;
        let text: String = if request.is_instance_of::<PyString>() {
            request.extract()?
        } else {
            let kw = PyDict::new(py);
            kw.set_item("ensure_ascii", false)?;
            json.call_method("dumps", (request,), Some(&kw))?
                .extract()?
        };
        let out = py
            .detach(|| self.engine().decide_json(&text))
            .map_err(to_py_err)?;
        json.call_method1("loads", (out,))
    }

    /// Same as `decide` with JSON text in and out.
    fn decide_json(&self, py: Python<'_>, request: &str) -> PyResult<String> {
        py.detach(|| self.engine().decide_json(request))
            .map_err(to_py_err)
    }

    /// Probabilities (after the temperature) of already rendered option texts, in order.
    ///
    /// `qtype` is `"choice"`, `"score"` or `"noul"`. The question and option tokens are cached,
    /// so calling this in a loop with a changing `state` only tokenizes the state.
    #[pyo3(signature = (qtype, instructions, options, state))]
    fn decide_raw(
        &self,
        py: Python<'_>,
        qtype: &str,
        instructions: &str,
        options: Vec<String>,
        state: &str,
    ) -> PyResult<Vec<f32>> {
        let qt = QType::from_name(qtype).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown question type {qtype:?} (choice, score, noul)"
            ))
        })?;
        py.detach(|| self.engine().decide_raw(qt, instructions, &options, state))
            .map_err(to_py_err)
    }

    /// Model description (architecture sizes, temperatures, ...) as a dict.
    #[getter]
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let text = self.engine().model_info().to_json();
        py.import("json")?.call_method1("loads", (text,))
    }

    /// Model name (`general.name`).
    #[getter]
    fn name(&self) -> String {
        self.engine().model_info().name
    }

    /// Number of parameters.
    #[getter]
    fn n_params(&self) -> usize {
        self.engine().n_params()
    }

    /// File the model was loaded from, or None when loaded from bytes.
    #[getter]
    fn path(&self) -> Option<String> {
        self.path.clone()
    }

    fn __repr__(&self) -> String {
        let info = self.engine().model_info();
        format!(
            "skycmd.Model(name={:?}, n_params={}, hidden_size={}, layers={})",
            info.name, info.n_params, info.hidden_size, info.num_layers
        )
    }
}

/// Load a model: same as `Model(source)`.
#[pyfunction]
fn load(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Model> {
    Model::new(py, source)
}

/// Tiny System One decision models: typed choice / score / yes-no probabilities in one forward pass.
#[pymodule]
fn skycmd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
