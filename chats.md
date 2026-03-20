



```mdx lerobot/docs/source/libero.mdx
# LIBERO

**LIBERO** is a benchmark designed to study **lifelong robot learning**. The idea is that robots won’t just be pretrained once in a factory, they’ll need to keep learning and adapting with their human users over time. This ongoing adaptation is called **lifelong learning in decision making (LLDM)**, and it’s a key step toward building robots that become truly personalized helpers.

- 📄 [LIBERO paper](https://arxiv.org/abs/2306.03310)
- 💻 [Original LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO)

To make progress on this challenge, LIBERO provides a set of standardized tasks that focus on **knowledge transfer**: how well a robot can apply what it has already learned to new situations. By evaluating on LIBERO, different algorithms can be compared fairly and researchers can build on each other’s work.

LIBERO includes **five task suites**:

- **LIBERO-Spatial (`libero_spatial`)** – tasks that require reasoning about spatial relations.
- **LIBERO-Object (`libero_object`)** – tasks centered on manipulating different objects.
- **LIBERO-Goal (`libero_goal`)** – goal-conditioned tasks where the robot must adapt to changing targets.
- **LIBERO-90 (`libero_90`)** – 90 short-horizon tasks from the LIBERO-100 collection.
- **LIBERO-Long (`libero_10`)** – 10 long-horizon tasks from the LIBERO-100 collection.

Together, these suites cover **130 tasks**, ranging from simple object manipulations to complex multi-step scenarios. LIBERO is meant to grow over time, and to serve as a shared benchmark where the community can test and improve lifelong learning algorithms.

![An overview of the LIBERO benchmark](https://libero-project.github.io/assets/img/libero/fig1.png)

# Evaluating with LIBERO-PRO

Add this to LeRobot’s pyproject.toml under the extras section: i.e inside optional-dependencies `# Optional dependencies
[project.optional-dependencies]`

you can find libero and other simulations listed. just add it there.

libero-pro = ["lerobot[transformers-dep]","libero @ git+https://github.com/stepanfeduniak/lerobot-libero-pro.git"]

Then install from your local LeRobot:

pip install -e ".[libero-pro]"


to check if its indeed installed - 

python - <<'PY'
import importlib, importlib.metadata as md, json
print("module file:", importlib.import_module("libero").__file__)
print("direct_url.json:", md.distribution("libero").read_text("direct_url.json"))
PY
  
to evaluate using libero-pro - 

```bash
lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --env.type=libero \
    --env.task=libero_object \
    --eval.batch_size=2 \
    --eval.n_episodes=3
```
Surprising - no change in script to run 

## Evaluating with LIBERO

At **LeRobot**, we ported [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) into our framework and used it mainly to **evaluate [SmolVLA](https://huggingface.co/docs/lerobot/en/smolvla)**, our lightweight Vision-Language-Action model.

LIBERO is now part of our **multi-eval supported simulation**, meaning you can benchmark your policies either on a **single suite of tasks** or across **multiple suites at once** with just a flag.

To Install LIBERO, after following LeRobot official instructions, just do:
`pip install -e ".[libero]"`

# Ensure your lerobot environment is active
conda activate lerobot

# Uninstall the potentially conflicting cmake Python package
pip uninstall cmake

# (Optional) Verify that your system cmake is now discoverable
which cmake # Should point to /usr/bin/cmake or similar system path
cmake --version # Should show your system cmake version (e.g., 3.22.1)

# Retry the libero installation
pip install -e ".[libero]"

### Single-suite evaluation

Evaluate a policy on one LIBERO suite:

Before this we had to pip install num2words

and

# ```export MUJOCO_GL=egl```

```bash
lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=2 \
  --eval.n_episodes=3
```

```bash
lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero \
  --env.task=libero_spatial,libero_object \
  --eval.batch_size=2 \
  --eval.n_episodes=250
```

- `--env.task` picks the suite (`libero_object`, `libero_spatial`, etc.).
- `--env.task_ids` picks task ids to run (`[0]`, `[1,2,3]`, etc.). Omit this flag (or set it to `null`) to run all tasks in the suite.
- `--eval.batch_size` controls how many environments run in parallel.
- `--eval.n_episodes` sets how many episodes to run in total.

---

### Multi-suite evaluation

Benchmark a policy across multiple suites at once:

```bash
lerobot-eval \
  --policy.path="your-policy-id" \
  --env.type=libero \
  --env.task=libero_object,libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=2
```

- Pass a comma-separated list to `--env.task` for multi-suite evaluation.

### Control Mode

LIBERO now supports two control modes: relative and absolute. This matters because different VLA checkpoints are trained with different mode of action to output hence control parameterizations.
You can switch them with: `env.control_mode = "relative"` and `env.control_mode = "absolute"`

### Policy inputs and outputs

When using LIBERO through LeRobot, policies interact with the environment via **observations** and **actions**:

- **Observations**
  - `observation.state` – proprioceptive features (agent state).
  - `observation.images.image` – main camera view (`agentview_image`).
  - `observation.images.image2` – wrist camera view (`robot0_eye_in_hand_image`).

  ⚠️ **Note:** LeRobot enforces the `.images.*` prefix for any multi-modal visual features. Always ensure that your policy config `input_features` use the same naming keys, and that your dataset metadata keys follow this convention during evaluation.
  If your data contains different keys, you must rename the observations to match what the policy expects, since naming keys are encoded inside the normalization statistics layer.
  This will be fixed with the upcoming Pipeline PR.

- **Actions**
  - Continuous control values in a `Box(-1, 1, shape=(7,))` space.

We also provide a notebook for quick testing:
Training with LIBERO

## Training with LIBERO

When training on LIBERO tasks, make sure your dataset parquet and metadata keys follow the LeRobot convention.

The environment expects:

- `observation.state` → 8-dim agent state
- `observation.images.image` → main camera (`agentview_image`)
- `observation.images.image2` → wrist camera (`robot0_eye_in_hand_image`)

⚠️ Cleaning the dataset upfront is **cleaner and more efficient** than remapping keys inside the code.
To avoid potential mismatches and key errors, we provide a **preprocessed LIBERO dataset** that is fully compatible with the current LeRobot codebase and requires no additional manipulation:
👉 [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero)

For reference, here is the **original dataset** published by Physical Intelligence:
👉 [physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero)

---

### Example training command

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id=${HF_USER}/libero-test \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir=./outputs/ \
  --steps=100000 \
  --batch_size=4 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq=1000 \
```

---

### Note on rendering

LeRobot uses MuJoCo for simulation. You need to set the rendering backend before training or evaluation:

- `export MUJOCO_GL=egl` → for headless servers (e.g. HPC, cloud)

## Reproducing π₀.₅ results

We reproduce the results of π₀.₅ on the LIBERO benchmark using the LeRobot implementation. We take the Physical Intelligence LIBERO base model (`pi05_libero`) and finetune for an additional 6k steps in bfloat16, with batch size of 256 on 8 H100 GPUs using the [HuggingFace LIBERO dataset](https://huggingface.co/datasets/HuggingFaceVLA/libero).

The finetuned model can be found here:

- **π₀.₅ LIBERO**: [lerobot/pi05_libero_finetuned](https://huggingface.co/lerobot/pi05_libero_finetuned)

We then evaluate the finetuned model using the LeRobot LIBERO implementation, by running the following command:

```bash
lerobot-eval \
  --output_dir=/logs/ \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.path=pi05_libero_finetuned \
  --policy.n_action_steps=10 \
  --output_dir=./eval_logs/ \
  --env.max_parallel_tasks=1
```

**Note:** We set `n_action_steps=10`, similar to the original OpenPI implementation.

### Results

We obtain the following results on the LIBERO benchmark:

| Model    | LIBERO Spatial | LIBERO Object | LIBERO Goal | LIBERO 10 | Average  |
| -------- | -------------- | ------------- | ----------- | --------- | -------- |
| **π₀.₅** | 97.0           | 99.0          | 98.0        | 96.0      | **97.5** |

These results are consistent with the original [results](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero#results) reported by Physical Intelligence:

| Model    | LIBERO Spatial | LIBERO Object | LIBERO Goal | LIBERO 10 | Average   |
| -------- | -------------- | ------------- | ----------- | --------- | --------- |
| **π₀.₅** | 98.8           | 98.2          | 98.0        | 92.4      | **96.85** |
```


as you can see we installed libero pro from that repo there
and it was named libero itself

We did this how to check if when i run the eval script is uses libero pro and libero ? because im getting same results in both cases which is suspicious. 
also check if we can clear the cache maybe i dont know.

also there is no disable video flag but i dont want video to be saved. and when i run huge script video is saved and system runs out of mem so find a soln for that. 

also when run there were exactly 10 tasks in libero pro as well but LIBERO-OBJECT has 10 base tasks. LIBERO-PRO-OBJECT keeps the same 10 base tasks but evaluates each under five additional perturbation settings (object attribute, position, semantic instruction, task-level instruction, and environment), yielding 50 perturbed variants and 60 total evaluation cases per object benchmark task set. 

Now I have files of the repo we installed libero pro from. lets do this in a rigorous manner lets see if libero eval scripts are using libero pro or libero and then lets dive into libero pro repo if in case we are using libero pro to see wahts happens how many tasks are there and why im getting same results in both. also before all this lets check if its a cache issue and re run 

I'll give a concise, practical plan you can run now to (A) determine whether the eval is using the libero-pro variant you installed, (B) clear any cache and reinstall deterministically, (C) confirm how many tasks / perturbations the active package exposes, and (D) stop videos from being saved (so you don't run out of memory). For each step I give exact commands / small Python snippets you can paste and run in your lerobot environment.

Important: run these commands inside the same conda env where you run lerobot-eval (e.g., conda activate lerobot).

1) Quick confirmation: which "libero" package is being imported by Python and which lerobot-eval binary is used
- Show the lerobot-eval binary path and its Python interpreter:
  which lerobot-eval
  head -n 1 $(which lerobot-eval)   # shows shebang, which Python it will use

- In that same env, run this Python snippet (paste exactly):
  python - <<'PY'
  import importlib, importlib.metadata as md, sys, json, os
  print("Python sys.executable:", sys.executable)
  m = importlib.import_module("libero")
  print("libero module file:", getattr(m, "__file__", "<no __file__>"))
  try:
      dist = md.distribution("libero")
      print("libero metadata location:", dist.locate_file(""))
      print("libero version:", dist.version)
      try:
          print("direct_url.json:", dist.read_text("direct_url.json"))
      except Exception as e:
          print("direct_url.json: not found or error:", e)
      try:
          print("METADATA head lines:")
          print("\\n".join(dist.read_text("METADATA").splitlines()[:20]))
      except Exception as e:
          print("METADATA: not found or error:", e)
  except Exception as e:
      print("importlib.metadata.distribution error:", e)
  # try printing top-level package repr to see clues
  print("module repr:", repr(m)[:400])
  # try to find any attribute referring to 'pro' or git url
  for name in dir(m):
      if name.lower().count("pro"):
          print("potential pro attr:", name, getattr(m, name))
  PY

What to look for:
- If module file points into a path you recognize (your local clone of the libero-pro repo), you're importing the installed editable copy. If it points into site-packages/libero from the pip-installed package, it will show that.
- direct_url.json (if present) will tell you the Git URL and commit if the package was installed from VCS. That will confirm whether the installed "libero" is actually the libero-pro repo (it should show the git+ URL or the fork you used).

2) If you want an even stricter check for editable install (git commit)
- cd to the directory printed by the module file (the repo root) and run:
  git rev-parse HEAD
  git remote -v
This confirms commit id and remote used.

3) Inspect which code path lerobot-eval is calling for env.type=libero
- Grep for the string that maps environment types into env constructors. From lerobot code, the CLI will eventually import the package/module that implements the "libero" environment. Easiest is to search your codebase and site-packages for "libero" strings:
  # replace /path/to/site-packages with the site-packages parent you saw above (or `python -c "import site; print(site.getsitepackages())"`)
  grep -R --line-number "env.type.*libero" .
  grep -R --line-number "register.*libero" $(python -c "import site, json, sys; print('\\n'.join(site.getsitepackages()))") 2>/dev/null || true

- Alternatively, run lerobot-eval with an increased logging level to see importer/debug info (if the CLI supports it):
  lerobot-eval --log.level=debug ...your-other-args...
Look in the debug output for paths and modules being imported when the environment is created.

4) If you see the wrong package or need a clean reinstall (clear caches)
- Uninstall the current libero package and clear pip cache:
  pip uninstall -y libero
  pip cache purge
- Remove any leftover egg-info and __pycache__ for libero in site-packages:
  python - <<'PY'
  import site, os, shutil, glob
  for p in site.getsitepackages():
      for f in glob.glob(os.path.join(p, "libero*")):
          print("removing", f)
          try:
              shutil.rmtree(f)
          except:
              try:
                  os.remove(f)
              except:
                  pass
  PY
- Reinstall explicitly from the repo you want:
  # editable install from local clone of libero-pro repo:
  pip install -e /path/to/your/libero-pro-clone
  # or from git URL (as you used before)
  pip install -e "git+https://github.com/stepanfeduniak/lerobot-libero-pro.git#egg=libero"

- Re-run the snippet from step (1) to confirm the module file and direct_url.json match the expected repo/commit.

5) Programmatically list tasks / perturbations exposed by that installed package
- Many libs expose a list of available tasks; if you can import the env class, print the tasks. Try this generic snippet (works if package has task lists or suites exposed):
  python - <<'PY'
  import importlib, json
  m = importlib.import_module("libero")
  print("module file:", m.__file__)
  # Heuristic: look for variables that list tasks
  for name in dir(m):
      if name.lower().startswith("libero") or "task" in name.lower() or "suite" in name.lower() or "variants" in name.lower():
          try:
              val = getattr(m, name)
              # only print short reprs
              s = repr(val)
              if len(s) > 400:
                  s = s[:400] + "..."
              print(name, ":", s)
          except Exception as e:
              print("skipping", name, "err", e)
  # If the package exposes a function to create an env, try to instantiate and query tasks.
  try:
      # best-effort: try to import the environment factory from lerobot if available
      import lerobot
      print("lerobot version", getattr(lerobot, "__version__", "<no version>"))
      # Many environments provide metadata: check for libero-specific helper
      try:
          import libero.envs as envs
          print("libero.envs attributes:", dir(envs)[:50])
      except Exception as e:
          pass
  except Exception:
      pass
  PY

- If that doesn't show a clear task count, create an env and inspect attributes. Example using gym-like API if the package exposes it (you may need to adjust to your framework's API):
  python - <<'PY'
  import importlib
  m = importlib.import_module("libero")
  # Try to find a function/class named make or create
  candidates = [getattr(m, n) for n in dir(m) if n.lower() in ("make", "create_env", "create_environment")]
  print("candidates:", [c.__name__ for c in candidates if hasattr(c, "__name__")])
  # If the repo contains a registry file, manually inspect its task list on disk:
  import inspect, os
  print("module file dir:", os.path.dirname(m.__file__))
  PY

If you prefer, point me to the repo files you mentioned and I can search them (but you said you already have them locally).

6) Why you might be getting the same results for "libero" and "libero-pro"
- If both CLI invocations import the same "libero" package (because libero-pro was installed under the name "libero"), they will behave identically — that would explain identical results.
- Another possibility: libero-pro is a thin wrapper that reuses the same task logic and default seeds as base LIBERO in many evaluation cases; only the perturbation generator is enabled under a flag. So if you run the suite that only exercises base tasks (libero_object), you may get the same results unless you explicitly enable perturbed variants. Check the libero-pro README or package code for flags like enable_perturbations=True or perturbation_modes=[...].
- Confirm by exploring the installed repo: grep for "perturb", "perturbation", "attribute", "semantic", "instruction" in the installed package:
  grep -R --line-number -E "perturb|perturbation|attribute|semantic|instruction" $(python -c "import importlib; print(importlib.import_module('libero').__file__)")/.. 2>/dev/null || true

7) Prevent videos from being saved (several approaches — try them in order)
A. Preferred: CLI flag or env setting (fast to try)
- Check if lerobot-eval supports a flag to disable video. Run:
  lerobot-eval --help | grep -i video || true
If there is an --env.record_video or --no-video flag, use it.

B. Lower memory/disable video via output_dir
- Point output_dir to a tmpfs with limited size or /dev/shm carefully OR to a location with small quota:
  --output_dir=/tmp/lerobot_eval_novideo
- Or set output_dir to a RAM-limited partition so if large video writing starts it fails early. Not ideal but can protect disk.

C. Monkeypatch the recorder to no-op (reliable)
- Find the video recording implementation in the lerobot/libero or in the evaluation runner (grep for RecordVideo, VideoWriter, ffmpeg, imageio.get_writer). Example:
  grep -R --line-number -E "RecordVideo|VideoWriter|imageio.get_writer|ffmpeg" $(python -c "import importlib, os; print(os.path.dirname(importlib.import_module('libero').__file__))") 2>/dev/null || true
- Edit that file in your local installed repo to short-circuit recording. For example, if you find a class VideoRecorder with a record(frame) method, change it to do nothing. Because your install is editable (recommended), this is safe and easy.

D. If you cannot edit installed code, create a tiny wrapper console script that imports lerobot-eval’s runner and patches the recorder before the evaluation starts:
  python - <<'PY'
  import importlib
  # import the module that creates/uses a recorder and monkeypatch its writer
  # Example (pseudocode): 
  # import lerobot.eval.runner as runner
  # runner.VideoRecorder.record = lambda *a, **k: None
  # then call the actual runner entrypoint function
  PY
This is more advanced and requires locating the right symbols.

E. Reduce batch size / parallelism
- Videos may be multiplied by parallel envs. Reduce:
  --eval.batch_size=1 --env.max_parallel_tasks=1
This reduces memory and I/O pressure.

F. Final defensive approach: make the videos directory unwritable for the process so the saver cannot write them:
  chmod -w /path/to/eval_logs/videos
This may cause exceptions if code doesn't handle write failures gracefully so watch logs.

8) If you want, I can examine the repo files you have
- If you paste (or point me to) the top-level filenames or the output of:
  ls -la /path/to/your/libero-pro-clone
  grep -R --line-number -E "perturb|perturbation|record|video|RecordVideo|VideoWriter|imageio|ffmpeg" /path/to/your/libero-pro-clone | head -n 200
I can point exactly which flag to flip to enable perturbed evaluations and where to disable video recording.

9) Short checklist you can run now to get to a clean experiment
- Run step (1) snippet to confirm current installed "libero".
- If it points to the libero-pro repo, you are already using libero-pro — proceed to step (6) to find perturbation flags.
- If it does not, uninstall and reinstall explicitly from the repo (step 4).
- To stop videos immediately: run eval with --eval.batch_size=1 --env.max_parallel_tasks=1 and set output_dir to a temp dir; then search the code for RecordVideo and monkeypatch to no-op.

If you want, paste the outputs of these two commands (from the environment you run lerobot-eval in):
- python - <<'PY'
  import importlib, importlib.metadata as md, json
  m = importlib.import_module("libero")
  print("libero module file:", m.__file__)
  try:
      print("direct_url.json:", md.distribution("libero").read_text("direct_url.json"))
  except Exception as e:
      print("direct_url.json not available:", e)
  print("site-packages location(s):", importlib.util.find_spec("libero").origin)
  PY

- which lerobot-eval; head -n 1 $(which lerobot-eval)

Paste those outputs here and I will tell you exactly whether you are running libero-pro and where the perturbations and video recorder live so you can turn them off.

dont do all at once

this is some outputs lerobot) dhruv@iiitb-X570-AORUS-ELITE-WIFI:~/lerobot$ which lerobot-eval
/home/dhruv/miniforge3/envs/lerobot/bin/lerobot-eval
(lerobot) dhruv@iiitb-X570-AORUS-ELITE-WIFI:~/lerobot$ head -n 1 $(which lerobot-eval)   # shows shebang, which Python it will use
#!/home/dhruv/miniforge3/envs/lerobot/bin/python3.10
(lerobot) dhruv@iiitb-X570-AORUS-ELITE-WIFI:~/lerobot$ python
Python 3.10.19 | packaged by conda-forge | (main, Jan 26 2026, 23:45:08) [GCC 14.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import importlib, importlib.metadata as md, sys, json, os
>>> print("Python sys.executable:", sys.executable)
Python sys.executable: /home/dhruv/miniforge3/envs/lerobot/bin/python
>>> m = importlib.import_module("libero")
>>> print("libero module file:", getattr(m, "__file__", "<no __file__>"))
libero module file: /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/__init__.py
>>> dist = md.distribution("libero")
>>> print("libero metadata location:", dist.locate_file(""))
libero metadata location: /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages
>>> print("libero version:", dist.version)
libero version: 0.1.1
>>> print("direct_url.json:", dist.read_text("direct_url.json"))
direct_url.json: {"url": "https://github.com/stepanfeduniak/lerobot-libero-pro.git", "vcs_info": {"commit_id": "99fb99f8a7c3acb1955c7b922eb8ff394433a4a9", "vcs": "git"}}
>>> print("METADATA head lines:")
METADATA head lines:
>>> print("\\n".join(dist.read_text("METADATA").splitlines()[:20]))
Metadata-Version: 2.4\nName: libero\nVersion: 0.1.1\nSummary: LIBERO: Benchmark and environments for robot learning\nAuthor: Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone\nAuthor-email: Jade Choghari <jade.choghari@huggingface.co>\nLicense: MIT\nProject-URL: Homepage, https://github.com/huggingface/lerobot-libero\nProject-URL: Issues, https://github.com/huggingface/lerobot-libero/issues\nRequires-Python: >=3\nDescription-Content-Type: text/markdown\nLicense-File: LICENSE\nRequires-Dist: hf-egl-probe @ git+https://github.com/huggingface/egl_probe.git#egg=hf-egl-probe\nRequires-Dist: transformers<5.0.0,>=4.53.0\nRequires-Dist: hydra-core<1.4,>=1.2\nRequires-Dist: robomimic==0.2.0\nRequires-Dist: robosuite==1.4.0\nRequires-Dist: bddl==1.0.1\nRequires-Dist: numpy\nRequires-Dist: transformers
>>> print("module repr:", repr(m)[:400])
module repr: <module 'libero' from '/home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/__init__.py'>
>>>

Good — thanks for the outputs. Short summary first, then the single next action to run.

What your output shows
- The installed libero package is coming from the libero-pro repo:
  - direct_url.json points to https://github.com/stepanfeduniak/lerobot-libero-pro.git (commit 99fb99f…).
  - The module file is /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/__init__.py
- That means lerobot-eval will import the package named "libero" from that installation. In other words, “libero” and your installed “libero-pro” are the same Python package in this environment — so running eval with env.type=libero uses that installed code. That alone can explain why results are identical between your two runs (they were both loading the same package).

Next step (one command)
- Let’s now search that installed package for the code that implements perturbations (the additional perturbed variants in libero-pro) and for any video/recorder code so we can later enable perturbations or disable video recording. Run this one Python snippet in the same environment and paste back the output:

python - <<'PY'
import importlib, os, re, sys
pkg = importlib.import_module("libero")
pkg_dir = os.path.dirname(pkg.__file__)
print("libero package dir:", pkg_dir)
keywords = r'perturb|perturbation|perturbations|attribute|semantic|instruction|variant|variants|num_variants|record|RecordVideo|VideoWriter|imageio|ffmpeg|video'
for root,dirs,files in os.walk(pkg_dir):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            try:
                s = open(path, "r", errors="ignore").read()
            except Exception:
                continue
            if re.search(keywords, s, re.I):
                print("---", path)
                for m in re.finditer(r'.{0,80}(' + keywords + r').{0,80}', s, re.I):
                    print("   ...", m.group(0).strip().replace("\n"," "))
                print()
PY

Why this next step
- That will tell us the exact files and code paths handling perturbations and video recording in the installed package. Once you paste the output I will:
  - Point to the exact flag(s) / function(s) that enable perturbed evaluations (so you can run the 50 variants per task).
  - Show where to disable video recording (quick edits / monkeypatchs / CLI flags to avoid saving video and running out of memory).
  - If needed, give the safe reinstall / cache-clearing commands next.

Run that snippet and paste the printed results — then I’ll give the focused next actions (enable perturbations or disable video) based on the actual files found.

libero package dir: /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero
--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/libero/utils/dataset_utils.py
   ... language_instruction = "".join(problem_info["language_instruction"])
   ... print("language instruction: {}".format(language_instruction.strip('"')))

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/libero/utils/video_utils.py
   ... import imageio
   ... class VideoWriter:
   ... def __init__(self, video_path, save_video=False, fps=30, single_video=True):
   ... self.video_path = video_path
   ... self.save_video = save_video
   ... self.single_video = single_video
   ... """Directly append an image to the video."""
   ... if self.save_video:
   ... """Append a camera observation to the video."""
   ... if self.save_video:
   ... if self.save_video:
   ... if self.save_video:
   ... if self.save_video:
   ... os.makedirs(self.video_path, exist_ok=True)
   ... if self.single_video:
   ... video_name = os.path.join(self.video_path, f"video.mp4")
   ... video_writer = imageio.get_writer(video_name, fps=self.fps)
   ... video_writer.append_data(im)
   ... video_writer.close()
   ... video_name = os.path.join(self.video_path, f"{idx}.mp4")
   ... video_writer = imageio.get_writer(video_name, fps=self.fps)
   ... video_writer.append_data(im)
   ... video_writer.close()
   ... print(f"Saved videos to {self.video_path}.")

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/libero/utils/log_utils.py
   ... def format(self, record):
   ... log_fmt = self.FORMATS.get(record.levelno)
   ... return formatter.format(record)

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/libero/envs/utils.py
   ... # multiply this quat by the object's initial rotation if it has the attribute specified

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/libero/envs/bddl_utils.py
   ... for attribute in region[1:]:
   ... if attribute[0] == ":target":
   ... assert len(attribute) == 2
   ... region_dict["target"] = attribute[1]
   ... target_name = attribute[1]
   ... elif attribute[0] == ":ranges":
   ... for rect_range in attribute[1]:
   ... elif attribute[0] == ":yaw_rotation":
   ... # print(attribute[1])
   ... for value in attribute[1]:
   ... elif attribute[0] == ":rgba":
   ... len(attribute[1]) == 4
   ... issing specification for rgba color, supposed to be 4 dimension, but only got  {attribute[1]}"
   ... region_dict["rgba"] = [float(x) for x in attribute[1]]
   ... for attribute in region[1:]:
   ... if attribute[0] == ":floor":
   ... assert len(attribute) == 2
   ... scene_properties_dict["floor_style"] = attribute[1]
   ... elif attribute[0] == ":wall":
   ... assert len(attribute) == 2
   ... scene_properties_dict["wall_style"] = attribute[1]
   ... language_instruction = ""
   ... language_instruction = group
   ... "language_instruction": " ".join(language_instruction),
   ... language_instruction = ""
   ... language_instruction = group
   ... "language_instruction": language_instruction,

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/libero/envs/env_wrapper.py
   ... self.language_instruction = problem_info["language_instruction"]

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/libero/envs/venv.py
   ... except (AttributeError, NotImplementedError):
   ... # mp may be deleted so it may raise AttributeError
   ... except (BrokenPipeError, EOFError, AttributeError):
   ... def __getattribute__(self, key: str) -> Any:
   ... """Switch the attribute getter depending on the key.
   ... Any class who inherits ``gym.Env`` will inherit some attributes, like
   ... ``action_space``. However, we would like the attribute lookup to go straight
   ... return super().__getattribute__(key)
   ... """Get an attribute from the underlying environments.
   ... If id is an int, retrieve the attribute denoted by key from the environment
   ... element. Otherwise, retrieve the attribute for all workers at indices id and
   ... :param str key: The key of the desired attribute.
   ... :return list: The list of environment attributes.
   ... """Set an attribute in the underlying environments.
   ... If id is an int, set the attribute denoted by key from the environment
   ... Otherwise, set the attribute for all workers at indices id.
   ... :param str key: The key of the desired attribute.
   ... :param Any value: The new value of the attribute.

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/libero/envs/bddl_base_domain.py
   ... def language_instruction(self):

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/libero/envs/robots/on_the_ground_panda.py
   ... self.set_joint_attribute(

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/libero/envs/robots/mounted_panda.py
   ... self.set_joint_attribute(

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/libero/envs/regions/base_region_sampler.py
   ... # multiply this quat by the object's initial rotation if it has the attribute specified
   ... # multiply this quat by the object's initial rotation if it has the attribute specified
   ... # multiply this quat by the object's initial rotation if it has the attribute specified
   ... raise AttributeError(

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/libero/envs/regions/object_property_sampler.py
   ... # Setup attributes

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/libero/benchmark/__init__.py
   ... "libero_10_semantic_ood",
   ... "libero_goal_semantic_ood",
   ... "libero_spatial_semantic_ood",
   ... "libero_object_semantic_ood",
   ... class LIBERO_10_SEMANTIC_OOD(Benchmark):
   ... self.name = "libero_10_semantic_ood"
   ... class LIBERO_GOAL_SEMANTIC_OOD(Benchmark):
   ... self.name = "libero_goal_semantic_ood"
   ... class LIBERO_SPATIAL_SEMANTIC_OOD(Benchmark):
   ... self.name = "libero_spatial_semantic_ood"
   ... class LIBERO_OBJECT_SEMANTIC_OOD(Benchmark):
   ... self.name = "libero_object_semantic_ood"

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/libero/benchmark/libero_suite_task_map.py
   ... "libero_10_semantic_ood": [
   ... "libero_spatial_semantic_ood": [
   ... "libero_goal_semantic_ood": [
   ... "libero_object_semantic_ood": [

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/lifelong/metric.py
   ... from libero.libero.utils.video_utils import VideoWriter
   ... # record the sim states for replay purpose
   ... # a new form of success record

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/lifelong/datasets.py
   ... filter_by_attribute=filter_key,  # can optionally provide a filter key here

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/lifelong/evaluate.py
   ... from libero.libero.utils.video_utils import VideoWriter
   ... parser.add_argument("--save-videos", action="store_true")
   ... video_folder = os.path.join(
   ... rk}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}_videos",
   ... with Timer() as t, VideoWriter(video_folder, args.save_videos) as video_writer:
   ... video_writer.append_vector_obs(

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/lifelong/models/bc_transformer_policy.py
   ... class PerturbationAttention:
   ... See https://arxiv.org/pdf/1711.00138.pdf for perturbation-based visualization

--- /home/dhruv/miniforge3/envs/lerobot/lib/python3.10/site-packages/libero/lifelong/algos/packnet.py
   ... # we do not record the success rate

Good — thanks, this is clear and useful. Quick summary of what your output shows, and one focused next step you can run.

Short summary (what we learned)
- Your lerobot-eval is importing the installed "libero" package from site-packages, and that package was installed from the libero-pro repo (direct_url.json points to https://github.com/stepanfeduniak/lerobot-libero-pro.git). So both of your runs (the one you called "libero" and the one you called "libero-pro") are using the same installed package — that explains why the results were identical.
- The installed package contains both:
  - perturbed/semantic-OOD suite definitions (look under libero/benchmark and libero/benchmark/libero_suite_task_map.py — there are suite names like libero_object_semantic_ood, libero_spatial_semantic_ood, etc.), and
  - a VideoWriter implementation in libero/libero/utils/video_utils.py and a "--save-videos" argument in libre/lifelong/evaluate.py.
- This implies: to exercise the extra perturbations you must explicitly run the semantic_ood suites (or pass whatever flags the repo expects). And to avoid videos you can disable the VideoWriter or stop the code path that supplies save_videos=True.

One focused next action
- Try running an evaluation against one of the semantic-OOD (perturbed) suites to see different behavior (this will confirm that the installed package supports the perturbed variants). Run this command (single env, no parallelism to be safe):

lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object_semantic_ood \
  --eval.batch_size=1 \
  --eval.n_episodes=3 \
  --env.max_parallel_tasks=1 \
  --output_dir=./eval_logs_novideo


i want to know what other flags exist like 

lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object_semantic_ood \
  --eval.batch_size=1 \
  --eval.n_episodes=3 \
  --env.max_parallel_tasks=1 \
  --output_dir=./eval_logs_novideo
  
  here instead of ood what else could be there
  

Short answer: you can run any suite name the package exposes (base suites like libero_object, libero_spatial, etc.) and the libero-pro repo also exposes perturbed/OOD suite names (e.g. libero_object_semantic_ood). The exact available names are defined in the package — you can list them with the tiny Python snippets below.

Run these in your same lerobot env to see everything available and to inspect evaluation flags.

1) Print all available suite/task names the installed package exposes
This prints the suite keys defined in the library (exact names you can pass to --env.task):

python - <<'PY'
import importlib, os
try:
    m = importlib.import_module("libero.benchmark.libero_suite_task_map")
except Exception:
    m = importlib.import_module("libero.benchmark")
print("module:", m.__name__)
# try common variable names
for key in ("SUITES", "suite_map", "TASK_MAP", "TASKS"):
    if hasattr(m, key):
        print(key, "->", list(getattr(m, key).keys()) if isinstance(getattr(m, key), dict) else getattr(m, key))
# fallback: show names that look like suite identifiers in the module
names = [n for n in dir(m) if ("libero" in n or n.endswith("_ood") or n.startswith("libero_"))]
print("Candidate names from module dir():", sorted(set(names)))
# show files in benchmark dir for manual inspection
pkg_dir = os.path.dirname(importlib.import_module("libero").__file__)
bench_dir = os.path.join(pkg_dir, "libero", "benchmark")
print("benchmark dir:", bench_dir)
print("files:", [f for f in os.listdir(bench_dir) if f.endswith(".py")])
PY

2) If you prefer a simple grep to see explicit suite names
Run this to find literal suite names in the package code:

grep -R --line-number -E "libero_[a-z0-9_]+(_ood)?|_semantic_ood|_attribute_ood|_position_ood|_env_ood|_instruction_ood" \
  $(python - <<'PY'
import importlib, os
print(os.path.dirname(importlib.import_module("libero").__file__))
PY
)/libero 2>/dev/null | sed -n '1,200p'

3) Inspect the evaluation CLI flags in the lifelong evaluate script
This shows flags like --save-videos and other custom args used by the lifelong eval entrypoint:

# print the lines that add arguments in the evaluate script
python - <<'PY'
import importlib, inspect, os
p = importlib.import_module("libero.lifelong.evaluate")
src = inspect.getsource(p)
for line in src.splitlines():
    if "add_argument" in line:
        print(line.strip())
PY

4) Quick, practical expected names (from docs + what we found)
- Base suites (common):
  - libero_object
  - libero_spatial
  - libero_goal
  - libero_90 (or libero_90 / libero_100 variants)
  - libero_10 (long tasks)
- Perturbed / OOD suites (libero-pro adds these flavors):
  - libero_object_semantic_ood
  - libero_spatial_semantic_ood
  - libero_goal_semantic_ood
  - libero_10_semantic_ood
- There may be additional perturbed flavors for attribute, position, instruction, environment (libero-pro description you quoted mentions object attribute, position, semantic instruction, task-level instruction, environment). The exact suite-name suffixes are in the mapping file; run step (1) to get the authoritative list.

5) How to pass multiple suites / tasks
- You can pass a comma-separated list to --env.task, e.g.:
  --env.task=libero_object,libero_object_semantic_ood
- To run specific task ids inside a suite, use --env.task_ids (same as LeRobot docs).

6) Example: list and then run
- Run the snippet in (1) and paste its output here if you want — I’ll point to the exact names and which to run to get the five perturbation types you mentioned.
- If you want to avoid videos while testing, note the evaluate script defines --save-videos (default off). So avoid passing that flag and set --output_dir to a safe small folder. If videos still appear, the package may default to True in some codepaths — I can point you to the exact VideoWriter usage and how to monkeypatch it once you paste the suite list.

If you run the snippet in (1) and paste the output here I will: 
- highlight the exact suite names you can pass to --env.task, and
- show the exact flag(s) in the evaluate script (and the VideoWriter implementation) so you can ensure videos are not saved.
