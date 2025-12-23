/* eslint-disable no-restricted-globals */

// Load Pyodide
importScripts("https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js");

let pyodide = null;
let pythonModel = null;

let readyPromiseResolve;
let readyPromise = new Promise(resolve => {
  readyPromiseResolve = resolve;
});

async function loadPyodideAndPackages(pythonScriptUrl) {
  self.pyodide = await loadPyodide();
  await self.pyodide.loadPackage("numpy");

  // Fetch the python code
  const response = await fetch(pythonScriptUrl);
  const pythonCode = await response.text();

  // Load the module
  self.pyodide.runPython(pythonCode);

  // Initialize the model
  self.pyodide.runPython(`
    model = BayesianBeliefModel()
  `);

  console.log("Pyodide and model initialized");
  readyPromiseResolve();
}

self.onmessage = async (event) => {
  const { id, type, payload } = event.data;

  if (type === "INIT") {
    await loadPyodideAndPackages(payload.pythonScriptUrl);
    self.postMessage({ id, result: { status: "initialized" } });
    return;
  }

  await readyPromise;

  try {
    let result;

    switch (type) {
      case "GET_STATE":
        result = self.pyodide.runPython(`
          import json
          state = model.get_state()
          # Convert to JSON string to avoid proxy issues with complex objects
          json.dumps(state)
        `);
        result = JSON.parse(result);
        break;

      case "ADD_EXPERIENCE":
        self.pyodide.globals.set("value", payload.value);
        self.pyodide.globals.set("sigma", payload.sigma);
        result = self.pyodide.runPython(`
          exp = model.add_experience(value, sigma)
          json.dumps(exp)
        `);
        result = JSON.parse(result);
        break;

      case "MEDITATE":
        self.pyodide.globals.set("level", payload.meditation_level);
        result = self.pyodide.runPython(`
          model.meditate_on_latest(level)
          json.dumps({"status": "success"})
        `);
        result = JSON.parse(result);
        break;

      case "SET_BRAHMA_VIHARAS":
        self.pyodide.globals.set("level", payload.level);
        result = self.pyodide.runPython(`
          model.set_brahma_viharas_level(level)
          json.dumps({"status": "success"})
        `);
        result = JSON.parse(result);
        break;

      case "RESET":
        self.pyodide.runPython(`
          model = BayesianBeliefModel()
        `);
        result = { status: "success" };
        break;

      default:
        throw new Error(`Unknown message type: ${type}`);
    }

    self.postMessage({ id, result });

  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
