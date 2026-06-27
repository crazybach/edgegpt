# CPU Setup

This checkout is set up for a CPU-only PyTorch environment in `.venv`.

## Activate

```powershell
cd C:\workspace\edgegpt
.\.venv\Scripts\Activate.ps1
```

If `python` is not on `PATH`, use the venv executable directly:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

## Recreate The Environment

```powershell
cd C:\workspace\edgegpt
C:\Users\crazy\AppData\Local\Programs\Python\Python313\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Verify CPU PyTorch

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available())"
```

Expected here:

```text
2.12.1+cpu
cuda False
```

## CPU Config

Use `configs/cpu.yaml` for future training or experiments on this machine. It
forces `device: "cpu"`, uses `dtype: "fp32"`, and lowers the model/context/batch
sizes while keeping `vocab_size: 16384` compatible with the checked-in tokenizer.
