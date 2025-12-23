(function(){"use strict";importScripts("https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js");let l,d=new Promise(s=>{l=s});async function i(s){self.pyodide=await loadPyodide(),await self.pyodide.loadPackage("numpy");const o=await(await fetch(s)).text();self.pyodide.runPython(o),self.pyodide.runPython(`
    model = BayesianBeliefModel()
  `),console.log("Pyodide and model initialized"),l()}self.onmessage=async s=>{const{id:a,type:o,payload:t}=s.data;if(o==="INIT"){await i(t.pythonScriptUrl),self.postMessage({id:a,result:{status:"initialized"}});return}await d;try{let e;switch(o){case"GET_STATE":e=self.pyodide.runPython(`
          import json
          state = model.get_state()
          # Convert to JSON string to avoid proxy issues with complex objects
          json.dumps(state)
        `),e=JSON.parse(e);break;case"ADD_EXPERIENCE":self.pyodide.globals.set("value",t.value),self.pyodide.globals.set("sigma",t.sigma),e=self.pyodide.runPython(`
          exp = model.add_experience(value, sigma)
          json.dumps(exp)
        `),e=JSON.parse(e);break;case"MEDITATE":self.pyodide.globals.set("level",t.meditation_level),e=self.pyodide.runPython(`
          model.meditate_on_latest(level)
          json.dumps({"status": "success"})
        `),e=JSON.parse(e);break;case"SET_BRAHMA_VIHARAS":self.pyodide.globals.set("level",t.level),e=self.pyodide.runPython(`
          model.set_brahma_viharas_level(level)
          json.dumps({"status": "success"})
        `),e=JSON.parse(e);break;case"RESET":self.pyodide.runPython(`
          model = BayesianBeliefModel()
        `),e={status:"success"};break;default:throw new Error(`Unknown message type: ${o}`)}self.postMessage({id:a,result:e})}catch(e){self.postMessage({id:a,error:e.message})}}})();
