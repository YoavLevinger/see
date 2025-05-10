# 🧠 How to Install and Run Ollama (`mistral`) on Ubuntu, macOS, and Windows

## 📦 Prerequisite: What is Ollama?
Ollama lets you run large language models (like Mistral) locally on your machine with simple commands.

---

## 🐧 Ubuntu

### ✅ Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

> **Note**: This installs Ollama to `/usr/local/bin/ollama`.

### ▶️ Run Mistral

```bash
ollama run mistral
```

### ▶️ Or Start Ollama Server

```bash
ollama serve
```

---

## 🍎 macOS (Intel & Apple Silicon)

### ✅ Install Ollama

Use Homebrew:

```bash
brew install ollama
```

Or use the official installer:

- Download from: [https://ollama.com/download](https://ollama.com/download)

### ▶️ Run Mistral

```bash
ollama run mistral
```

### ▶️ Or Start Ollama Server

```bash
ollama serve
```

---

## 🪟 Windows (via WSL or Native)

> **Note:** As of now, native Windows support is limited. The recommended method is **using WSL (Windows Subsystem for Linux)**.

### ✅ Install WSL (if not already installed)

```powershell
wsl --install
```

Then open your WSL terminal (Ubuntu), and follow the **Ubuntu** installation steps above.

### ▶️ Run Mistral

```bash
ollama run mistral
```

### ▶️ Or Start Ollama Server

```bash
ollama serve
```

> **Native Windows version:** Check [https://ollama.com](https://ollama.com) for updates as native support evolves.

---

## ✅ Verification

Once installed, you can verify everything works by running:

```bash
ollama list
```

You should see Mistral downloaded or listed after the first run.

---

