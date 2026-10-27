# Walkthrough: rebuild `genai` locally in VS Code

This rebuilds `ai2584987/genai` from scratch with all three assignments:

1. **Simple Text Generation API**: bigram `/generate` + spaCy `/embedding`
2. **CNN**: CIFAR-10 classifier at `/classify`
3. **GAN**: MNIST digit generator at `/generate_digits` (Assignment 3)

Do the steps in order. Each one ends with a ✅ check so you know it worked before moving on.

---

## Step 0: One-time setup (skip what you already have)

| Tool | Check in Terminal | Install if missing |
|---|---|---|
| VS Code | — | https://code.visualstudio.com |
| VS Code extensions | — | In VS Code: Extensions (⇧⌘X) → install **Python** and **Jupyter** (both by Microsoft) |
| uv | `uv --version` | `curl -LsSf https://astral.sh/uv/install.sh \| sh`, then open a new terminal |
| git | `git --version` | `xcode-select --install` |
| Docker Desktop | `docker --version` | https://www.docker.com/products/docker-desktop (open it once so the engine is running) |

✅ All four commands print a version.

---

## Step 1: Put the project in a fresh folder and open it

1. Unzip `genai.zip`. You get a folder called `genai/`.
2. Move it next to your old work, e.g. `~/Documents/Columbia - Fall 2026/Applied Gen AI/genai`.
   Keep your old folder for now. Don't copy anything from it into this one.
3. In VS Code: **File → Open Folder…** → choose the new `genai` folder.
4. Open the built-in terminal: **View → Terminal** (or ⌃`). It opens inside the project folder.

✅ `pwd` in the VS Code terminal ends in `/genai`, and `ls` shows `app  helper_lib  notebooks  pyproject.toml ...`

---

## Step 2: Create the environment (one command)

```bash
uv sync
```

This reads `pyproject.toml` + `uv.lock` and builds `.venv/` with everything: FastAPI, spaCy + `en_core_web_md`, PyTorch, torchvision, plus `ipykernel` / `matplotlib` for notebooks. The first run downloads PyTorch, so it takes a few minutes.

Then tell VS Code to use it: **⇧⌘P → "Python: Select Interpreter" → pick `./.venv/bin/python`** (it says *Recommended*).

✅ This prints a version and no error:

```bash
uv run python -c "import torch, spacy, fastapi; print(torch.__version__)"
```

### What's where (quick tour)

```
app/            FastAPI service: main.py has every endpoint
  models/       trained weights (cnn_cifar10.pth, gan_mnist_generator.pth)
helper_lib/     training code from the Module 4 + Module 6 class activities
notebooks/      01_text_generation, 02_cnn_cifar10, 03_gan_mnist
Dockerfile      builds the API image
pyproject.toml  the ONE dependency file (the old app/pyproject.toml is gone)
```

---

## Step 3: Assignment 1 (Simple Text Generation API)

Start the server in the VS Code terminal:

```bash
uv run fastapi dev app/main.py
```

Open http://127.0.0.1:8000/docs in your browser.

- **POST /generate** → *Try it out* → body `{"start_word": "bigram", "length": 5}` → *Execute*
- **POST /embedding** → body `{"word": "king"}` → you get `"dimensions": 300`
- Try `{"word": "qzxvbnm"}` → **404** (spaCy has no vector for a made-up word)

Leave the server running. Open a **second** terminal with the **+** in the terminal panel for the next steps.

**Notebook:** open `notebooks/01_text_generation.ipynb`. In the top right click **Select Kernel → Python Environments → .venv**, then **Run All**. Cell 3 calls the running server; the FLAN-T5 cells at the end are your prompting exercises, unchanged.

✅ `/generate` returns text, `/embedding` returns 300 numbers.

---

## Step 4: Assignment 2 (CNN on CIFAR-10)

The trained weights are already in `app/models/cnn_cifar10.pth`, so the endpoint works straight away.

- In `/docs`: **POST /classify** → *Try it out* → choose any `.jpg` / `.png` (a photo of a cat, car, plane…) → *Execute*
- Or from the second terminal:

```bash
curl -X POST http://127.0.0.1:8000/classify -F "file=@/path/to/cat.jpg"
```

Response looks like `{"filename": "cat.jpg", "predicted_class": "cat", "confidence": 0.72}`.

**Optional retrain:** `notebooks/02_cnn_cifar10.ipynb` → Run All. It downloads CIFAR-10 (~170 MB) into `data/`, trains 10 epochs with `helper_lib`, and overwrites `app/models/cnn_cifar10.pth`.

✅ `/classify` returns one of the 10 CIFAR-10 classes.

---

## Step 5: Assignment 3 (GAN on MNIST)

### 5a. Read the code against the PDF

Open `helper_lib/model.py` and scroll to **Assignment 3**. Every layer maps to the spec:

| PDF says | Code |
|---|---|
| Noise (B, 100) → FC to 7×7×128, reshape | `self.fc = nn.Linear(100, 7*7*128)`, then `.view(-1, 128, 7, 7)` |
| ConvTranspose2d 128→64, k4 s2 p1 → 14×14, BN + ReLU | `self.deconv1`, `self.bn1`, `F.relu` |
| ConvTranspose2d 64→1, k4 s2 p1 → 28×28, Tanh | `self.deconv2`, `torch.tanh` |
| Conv2d 1→64, k4 s2 p1 → 14×14, LeakyReLU(0.2) | `self.conv1`, `F.leaky_relu(…, 0.2)` |
| Conv2d 64→128, k4 s2 p1 → 7×7, BN + LeakyReLU(0.2) | `self.conv2`, `self.bn2` |
| Flatten → Linear → single real/fake probability | `self.flatten`, `self.fc = nn.Linear(128*7*7, 1)`, `torch.sigmoid` |

The Module 6 class-activity pieces:

- `helper_lib/model.py`: `get_model("GAN")` now works alongside FCNN / CNN / EnhancedCNN
- `helper_lib/trainer.py`: `train_gan(model, data_loader, criterion, optimizer, device, epochs)`
- `helper_lib/generator.py`: `generate_samples(model, device, num_samples)` plots a grid
- `helper_lib/data_loader.py`: `get_mnist_loader()` scales pixels to [-1, 1] to match Tanh

### 5b. Train it (notebook)

Open `notebooks/03_gan_mnist.ipynb` → kernel **.venv** → **Run All**.

1. The shape check prints `Generator out: [4, 1, 28, 28]` and `Discriminator out: [4, 1]`.
2. Training downloads MNIST (~12 MB) into `data/` and runs 10 epochs, printing D and G loss per epoch. On an M-series Mac it uses `mps` automatically.
3. `generate_samples` shows a grid of handwritten-looking digits.
4. The last cell saves `app/models/gan_mnist_generator.pth`.

Trained weights are already included, so this step is optional for the API to work. Run it anyway, since the training run is part of the assignment.

### 5c. Try the endpoint

The `fastapi dev` server reloads on code changes but **not** on new weight files. Stop it (⌃C) and start it again, then:

- `/docs` → **GET /generate_digits** → *Try it out* → `num_samples` = 16 → *Execute* → the image shows right in the page
- Or from the terminal:

```bash
curl "http://127.0.0.1:8000/generate_digits?num_samples=16&seed=7" -o digits.png && open digits.png
```

The same `seed` always gives the same image. Leave it out for new digits each time.

✅ You see a 4×4 grid of digits.

---

## Step 6: Docker

Stop the dev server (⌃C). Make sure Docker Desktop is running, then:

```bash
docker build -t genai-app .
docker run -p 8000:80 genai-app
```

The first build downloads PyTorch inside the image, so give it several minutes (the image is several GB).
Open http://127.0.0.1:8000/docs and repeat a quick check of all five endpoints.

✅ All endpoints respond from the container. Stop it with ⌃C.

---

## Step 7: Push to GitHub with a clean history

This **replaces** what's on `ai2584987/genai`. The first push below saves the old version as a branch called `archive-old-main`, so nothing is lost.

In the VS Code terminal, from the `genai` folder:

```bash
git init -b main
git add .
git status
```

Check the `git status` list **before** committing. It should **not** include `.venv/`, `data/`, `checkpoints/` or `__pycache__/` (`.gitignore` handles that). It **should** include both `.pth` files in `app/models/`.

```bash
git commit -m "Rebuild: text generation API, CIFAR-10 CNN, MNIST GAN (Assignments 1-3)"

git remote add origin https://github.com/ai2584987/genai.git
git fetch origin
git push origin origin/main:refs/heads/archive-old-main   # backup of the old repo
git push -u origin main --force                            # replace main with the clean rebuild
```

If git asks you to sign in, let VS Code's GitHub sign-in handle it (a browser window opens).
**Don't put a token inside a notebook.** The old repo had a commit cleaning one out, and the clean history means none of that comes along.

✅ On github.com/ai2584987/genai you see the new structure and README. Under *branches* you see `archive-old-main`. Once you're happy, you can delete that branch on GitHub.

From now on, commit from the VS Code terminal (or the Source Control panel, ⌃⇧G), not from `!git` cells in notebooks.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` | Close and reopen the VS Code terminal after installing uv |
| Notebook says *ModuleNotFoundError: helper_lib* | The kernel isn't `.venv`. Click the kernel name (top right) → `.venv`. The first cell of each notebook also moves to the repo root. |
| `uv sync` can't find a torch wheel | An Intel Mac. PyTorch ≥ 2.3 has no Intel-Mac builds. Tell me your Mac model and I'll pin a version that works. |
| `/generate_digits` returns 503 | `app/models/gan_mnist_generator.pth` is missing. Run notebook 03 to the end, then restart the server. |
| Port 8000 already in use | Another server is still running. ⌃C it, or use `--port 8001` |
| Docker build very slow | Normal the first time (PyTorch is large). Later builds reuse the cached layer unless `uv.lock` changes. |
