# PILOT — Traductor HPC Serial → Paralelo con MoL

Traductor automático de código C/C++ **serial** a versiones **paralelas** mediante modelos de lenguaje accedidos vía [OpenRouter](https://openrouter.ai/). Soporta dos modos de operación:

- **Backend único**: traduce directamente a un paradigma (OpenMP, MPI, AVX, Kokkos, Go, Google Highway o Vectorize).
- **Pipeline multi-etapa**: encadena paradigmas y, en los pipelines HPC, intercala etapas `Opt*` resueltas con **Mix of LLMs (MoL)**. Cada agente produce una variante; solo se acepta si mejora el tiempo de la etapa anterior al menos un 3 % y el auditor la da por legítima.

Los pipelines con etapas `Opt*` disponibles son:

- `OptSerial-AVX-OptAVX-OpenMP-OptOpenMP-MPI-OptMPI`
- `OptSerial-AVX-OptAVX-OpenMP-OptOpenMP` (el mismo, sin MPI)
- `OptSerial-MPI-OptMPI-OpenMP-OptOpenMP-AVX-OptAVX`

También siguen los pipelines de solo traducción `AVX-OpenMP-MPI` y `Vectorize-OpenMP-MPI`. El nombre del directorio en `prompts/pipelines/` es el orden de las etapas.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Dependencias](https://img.shields.io/badge/dependencias-requests%20%7C%20pydantic-green)
![Estado](https://img.shields.io/badge/estado-activo-brightgreen)

---

## ✨ Características principales

### Traducción

- Traduce ficheros `.c` / `.cpp` serial a **OpenMP**, **MPI**, **AVX/AVX-512**, **Kokkos**, **Go**, **Google Highway** o **Vectorize** con un solo comando.
- **Corrección iterativa** automática ante errores de compilación (hasta 5 intentos, conversación multi-turno).
- **Verificación funcional** activada por defecto (`--no-compare` la desactiva): compara el checksum numérico o, si no hay checksum, el stdout del binario serial frente al traducido. Si no cuadra, reintenta con el prompt de error de ejecución hasta agotar `MAX_RETRIES`.
- **Benchmark integrado**: 1 calentamiento y mediana de 11 ejecuciones. Si el programa imprime su tiempo, el speedup es el del kernel; el tiempo de proceso se informa como coste total (arranque, init y comunicación). Cada ejecución tiene un límite de 300 s.

### Pipeline

- El perfil de compilación es **acumulativo**. En cuanto una etapa ha introducido MPI, las etapas OpenMP y AVX posteriores se compilan con `mpicc`/`mpicxx` y los flags que correspondan (`-fopenmp`, `-march=native`).
- La referencia serial se captura una sola vez al inicio y se reutiliza en `--compare` y en MoL.
- Una etapa `Opt*` que no supera el margen conserva el fuente de entrada. En el resumen de tiempos esa etapa sale como `n.d.`: no hubo variante aceptada. La etapa siguiente sigue midiendo a partir del último resultado aceptado.

### Mix of LLMs (MoL) en etapas Opt\*

- Un agente por modelo de `--mol-models` (por defecto tres modelos distintos). No hay varias muestras del mismo modelo.
- Los agentes se lanzan **en paralelo**. Cada variante se compila, corrige, verifica y mide **en serie**, para no mezclar los benchmarks.
- La métrica es el tiempo de kernel si el programa lo imprime; si no, el tiempo de proceso. El auditor solo entra si la variante mejora el baseline al menos un **3 %** (`MOL_MIN_IMPROVEMENT`).
- El auditor (`--mol-audit-model`) comprueba que la mejora sea legítima (sin recortar iteraciones ni trabajo). Si rechaza el candidato, se prueba el siguiente. Si ninguno vale, se conserva el input.
- Cada variante tiene hasta `MOL_FIX_ATTEMPTS` (3) correcciones, de compilación y de ejecución juntas.

### Medición OpenMP y MPI

En las etapas `OpenMP`, `OptOpenMP`, `MPI` y `OptMPI` el harness mide dos colocaciones cuando hay hyperthreading y se queda con la más rápida:

- **A**: un hilo por CPU lógica (`OMP_PLACES=threads`, `OMP_PROC_BIND=close`).
- **B**: un hilo por núcleo físico (`OMP_PLACES=cores`, `OMP_PROC_BIND=spread`).

Sin SMT solo existe A. Esas líneas de lanzamiento se inyectan también en el prompt de la etapa.

En un pipeline con MPI el lanzamiento no usa `MPI_PROCS`. Hay **un rango por socket**: `ranks = nodos × sockets` y los hilos OpenMP de cada rango son los del socket (`cores × threads/core`). Los nodos salen de `SLURM_JOB_NUM_NODES` o `SLURM_NNODES`; fuera de Slurm se cuenta 1. El comando es Open MPI 4.1.1 (`mpirun -np … --map-by ppr:1:socket:PE=<cores por socket>`). Un nodo con un solo socket no es un objetivo MPI válido: el pipeline se detiene.

`MPI_PROCS` (defecto 4) solo afecta al **backend MPI suelto**, que lanza `mpirun -np $MPI_PROCS`.

### Inyección de hardware

- Ejecuta `lscpu` al inicio con `LC_ALL=C` y parsea cores, sockets, NUMA, cachés e ISA.
- Antepone `## Target machine` a los prompts. En MPI y OptMPI añade además el bloque de lanzamiento (nodos, rangos e hilos).
- Las etapas `Opt*` reciben el tiempo medido del input para que el modelo tenga un baseline numérico.
- Si `lscpu` no está disponible, el pipeline sigue sin ese bloque.

### Extensibilidad sin tocar código

- **Backends** descubiertos desde `prompts/backends/` (hace falta `translate.txt` y `fix_errors.txt`).
- **Pipelines** descubiertos desde `prompts/pipelines/`. El nombre del directorio, partido por `-`, es el orden.
- Un prompt en `prompts/pipelines/<pipeline>/<Etapa>/` sustituye al del backend genérico. `fix_execution_errors.txt` es opcional; si falta, esa etapa no puede corregir errores de ejecución.

---

## 🛠️ Tecnologías

| Capa | Tecnología |
|------|------------|
| Lenguaje | Python 3.10+ |
| API LLM | OpenRouter (`requests`, `pydantic`); opcional Hugging Face con `--hf` |
| Compilación serial / OptSerial | `gcc` / `g++` con `-O3` |
| Compilación OpenMP | `gcc` / `g++` con `-fopenmp -march=native` |
| Compilación MPI | `mpicc` / `mpicxx` |
| Compilación híbrida | `mpicc`/`mpicxx` con `-fopenmp` y/o `-march=native` según el perfil acumulado |
| Compilación AVX | `gcc` / `g++` con `-march=native` (o el compilador MPI si MPI ya está en el perfil) |
| Compilación Kokkos | `g++` + flags desde `.env` |
| Compilación Go | `go build` |
| Compilación Google Highway | `g++` con `-lhwy` |
| Ejecución MPI en pipeline | `mpirun` de Open MPI 4.1.1, un rango por socket |
| Hardware | `lscpu` con `LC_ALL=C`; nodos desde Slurm |
| MoL paralelo | `concurrent.futures.ThreadPoolExecutor` |

**Dependencias** (`requirements.txt`): `requests>=2.31.0`, `pydantic>=2`.

---

## 📋 Requisitos previos

### Runtime

- **Python** 3.10 o superior
- **uv** o **pip** para instalar dependencias

### Herramientas del sistema (según backend o pipeline)

| Backend | Herramientas necesarias |
|---------|-------------------------|
| OpenMP | `gcc` o `g++` con soporte OpenMP |
| MPI | `mpicc` / `mpicxx` y `mpirun` (Open MPI 4.1.1 en el lanzamiento del pipeline) |
| AVX / Vectorize | `gcc` / `g++` con AVX2 o AVX-512 |
| Kokkos | `g++` (C++17+), Kokkos instalado y configurado |
| Go | Toolchain Go (`go build`) |
| Google Highway | `g++` con la librería `libhwy` |

Un pipeline que incluye MPI, OpenMP y AVX necesita las tres toolchains. En un nodo de un solo socket ese pipeline se aborta al llegar a MPI.

### Cuenta externa

- Cuenta en [OpenRouter](https://openrouter.ai/) con API key válida.
- (Opcional) Cuenta en [Hugging Face](https://huggingface.co/) para `--hf`. El programa lee `HF_API_KEY`.

---

## 📦 Instalación

```bash
cd PILOT-TIMED-Pipeline

# Entorno con uv (el intérprete del proyecto puede ser 3.14)
uv venv --python 3.14
uv pip install -r requirements.txt

# O con pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Contenido mínimo de `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-tu-clave-aqui
```

Los ejemplos de abajo usan `uv run`, que toma el `.venv` del directorio. Con el entorno ya activado, `python3 main.py` es equivalente.

---

## ⚙️ Configuración

Variables leídas desde `.env`:

| Variable | Obligatoria | Descripción |
|----------|-------------|-------------|
| `OPENROUTER_API_KEY` | Sí (salvo `--hf`) | Clave de OpenRouter |
| `HF_API_KEY` | Con `--hf` | Clave de Hugging Face. El CLI no lee `HF_TOKEN` |
| `MPI_PROCS` | No | Número de rangos del backend MPI suelto (defecto `4`). El pipeline no la usa |
| `KOKKOS_CXX` | No\* | Compilador C++ para Kokkos |
| `KOKKOS_CXXFLAGS` | No\* | Flags de compilación Kokkos |
| `KOKKOS_LDFLAGS` | No\* | Flags de enlace Kokkos |

\* Necesarias si usas Kokkos y `kokkos_config` no está en el PATH.

Constantes en `translator/config/settings.py`:

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `MAX_RETRIES` | `5` | Correcciones máximas por etapa de traducción |
| `BENCHMARK_RUNS` | `11` | Ejecuciones medidas; se informa la mediana, tras un calentamiento |
| `EXECUTION_TIMEOUT_SECONDS` | `300` | Límite de cada ejecución |
| `CHECKSUM_RTOL` | `1e-6` | Tolerancia relativa del checksum (absoluta si la referencia es 0). Se cambia con `--rtol` |
| `MOL_MIN_IMPROVEMENT` | `0.03` | Mejora mínima para auditar y poder aceptar una variante |
| `MOL_FIX_ATTEMPTS` | `3` | Correcciones por variante MoL |
| `DEFAULT_MODEL` | `"gpt-oss-120b"` | Modelo de traducción por defecto |
| `DEFAULT_BACKEND_NAME` | `"OpenMP"` | Backend por defecto |
| `MOL_DEFAULT_AGENT_MODELS` | `qwen/qwen3-coder`, `deepseek-v4-flash`, `openai/gpt-oss-120b` | Agentes MoL |
| `MOL_DEFAULT_AUDIT_MODEL` | `"anthropic/claude-sonnet-5"` | Auditor MoL |

---

## 🚀 Uso

### Modo backend único

```bash
# Backend y modelo por defecto (OpenMP + gpt-oss-120b). --compare va activado.
uv run main.py ruta/al/fichero.c

uv run main.py fichero.c -b MPI -m deepseek-r1
uv run main.py fichero.c -b AVX -m sonnet-5
uv run main.py fichero.c -b Kokkos -m qwen3-coder
uv run main.py fichero.c -b Go -m kimi-k2.6
uv run main.py fichero.c -b GHighway -m qwen3-coder
uv run main.py fichero.c -b Vectorize -m qwen3-coder

# Sin el bucle de correctitud
uv run main.py fichero.c -b OpenMP --no-compare

uv run main.py fichero.c -b MPI --metrics
```

### Modo pipeline

```bash
# Siete etapas: OptSerial → AVX → OptAVX → OpenMP → OptOpenMP → MPI → OptMPI
uv run main.py fichero.cpp \
  --pipeline OptSerial-AVX-OptAVX-OpenMP-OptOpenMP-MPI-OptMPI \
  -m gpt-oss-120b \
  --metrics

# El mismo orden, sin MPI
uv run main.py fichero.cpp --pipeline OptSerial-AVX-OptAVX-OpenMP-OptOpenMP

# Orden inverso: primero MPI, AVX al final
uv run main.py fichero.c --pipeline OptSerial-MPI-OptMPI-OpenMP-OptOpenMP-AVX-OptAVX

# Solo traducción, sin etapas Opt*
uv run main.py fichero.c --pipeline AVX-OpenMP-MPI
uv run main.py fichero.c --pipeline Vectorize-OpenMP-MPI

# Un solo agente, para una prueba corta. El auditor también puede ser un alias.
uv run main.py fichero.cpp \
  --pipeline OptSerial-AVX-OptAVX-OpenMP-OptOpenMP-MPI-OptMPI \
  -m gpt-oss-120b \
  --mol-models gpt-oss-120b \
  --mol-audit-model gpt-oss-120b \
  --metrics

# Reasoning en las etapas de traducción (no cambia el margen del 3 %)
uv run main.py fichero.c \
  --pipeline OptSerial-MPI-OptMPI-OpenMP-OptOpenMP-AVX-OptAVX \
  -m deepseek-r1 --thinking --thinking-effort high
```

### Descubrimiento

```bash
uv run main.py --list-backends
uv run main.py --list-pipelines
uv run main.py --list-models
```

### Flags extra y tolerancia

```bash
uv run main.py fichero.c -b OpenMP --flags "-DPOLYBENCH_TIME utilities/polybench.c"
uv run main.py fichero.c -b OpenMP --rtol 1e-4
```

### Scripts de lote

Hay que lanzarlos desde `PILOT-TIMED-Pipeline/`.

| Script | Qué hace |
|--------|----------|
| `run_pipeline_benchmarks.sh` | Pasa cada `.cpp` de `../../pipeline_benchmarks` por `OptSerial-AVX-OptAVX-OpenMP-OptOpenMP-MPI-OptMPI` con `gpt-oss-120b` como traductor y único agente, y `sonnet-5` como auditor. Tope de 3 h por benchmark. Logs en `logs/<fecha>_pipeline_benchmarks_<modelo>/`. |
| `run_new_benchmarks.sh` | Traduce un directorio de benchmarks (defecto `../../new_benchmarks_paper`) con un backend o un `--pipeline`. `--compare` es el defecto; `--no-compare` lo apaga. |
| `verify_benchmarks.sh` | Compila y compara fuentes ya traducidos, sin llamar al LLM. Delega en `verify_existing.py`. Hace falta `-m <modelo>`. |

```bash
./run_new_benchmarks.sh -b OpenMP -m qwen3-coder --metrics
./run_new_benchmarks.sh --pipeline AVX-OpenMP-MPI -m gpt-oss-120b
./verify_benchmarks.sh -b MPI -m sonnet-5
```

---

## 📖 Referencia de argumentos CLI

| Flag | Descripción |
|------|-------------|
| `source` | Fichero `.c` o `.cpp` a traducir |
| `-b`, `--backend` | Backend único (incompatible con `--pipeline`). Defecto: OpenMP |
| `--pipeline` | Pipeline multi-etapa por nombre de directorio (incompatible con `-b`) |
| `-m`, `--model` | Alias o ID de OpenRouter para las etapas de traducción. Defecto: `gpt-oss-120b` |
| `--list-backends` | Muestra backends detectados y sale |
| `--list-pipelines` | Muestra pipelines detectados y sale |
| `--list-models` | Muestra alias de modelos y sale |
| `--model-provider` | Fuerza un proveedor de OpenRouter (`provider.only`, p. ej. `Together`) |
| `--thinking` / `--no-thinking` | Activa o desactiva el reasoning. Desactivado por defecto |
| `--thinking-effort` | `minimal`, `low`, `medium`, `high` o `xhigh`. Requiere `--thinking` |
| `--flags` | Flags extra de compilación, en una sola cadena entre comillas |
| `--rtol` | Tolerancia del checksum. Defecto `1e-6` |
| `--compare` / `--no-compare` | Verificación de correctitud. Activada por defecto |
| `--hf` | Hugging Face en lugar de OpenRouter. Exige `HF_API_KEY` |
| `--debug` | JSON de cada petición y respuesta. Conserva los fuentes `_mol_v*` de las variantes |
| `--metrics` | Tokens, tiempo de inferencia y coste USD. Solo OpenRouter |
| `--mol-models` | Modelos agentes de las etapas `Opt*` (solo `--pipeline`) |
| `--mol-audit-model` | Modelo auditor (solo `--pipeline`) |

---

## 📁 Estructura del proyecto

```text
PILOT-TIMED-Pipeline/
├── main.py
├── verify_existing.py               # Verificación de fuentes ya traducidos
├── requirements.txt
├── .env.example
├── run_pipeline_benchmarks.sh
├── run_new_benchmarks.sh
├── verify_benchmarks.sh
├── prompts/
│   ├── backends/                    # Lo que descubre --list-backends
│   │   ├── AVX/  OpenMP/  MPI/  Vectorize/  Kokkos/  Go/  GHighway/
│   │   └── <Backend>/translate.txt  fix_errors.txt  [fix_execution_errors.txt]
│   ├── pipelines/
│   │   ├── AVX-OpenMP-MPI/
│   │   ├── Vectorize-OpenMP-MPI/
│   │   ├── OptSerial-AVX-OptAVX-OpenMP-OptOpenMP/
│   │   ├── OptSerial-AVX-OptAVX-OpenMP-OptOpenMP-MPI-OptMPI/
│   │   └── OptSerial-MPI-OptMPI-OpenMP-OptOpenMP-AVX-OptAVX/
│   └── OpenMP/ MPI/ Kokkos/ Go/ GHighway/   # Copias sueltas: el descubridor no las lee
└── translator/
    ├── exceptions.py
    ├── registry.py                  # Descubrimiento de backends y pipelines
    ├── cli/main.py
    ├── config/
    │   ├── settings.py              # Modelos, constantes, rutas, MoL
    │   └── env.py
    ├── domain/
    │   ├── backend.py
    │   └── pipeline.py              # ParadigmProfile, PipelineStage, Pipeline
    ├── services/
    │   ├── translate.py             # Modo backend único
    │   ├── stage_runner.py          # Una etapa de traducción
    │   ├── pipeline_runner.py       # Orquestación y resumen de tiempos
    │   ├── mol_runner.py            # Agentes, margen del 3 % y auditor
    │   ├── prompt_blocks.py         # Hardware, lanzamiento y baseline inyectados
    │   └── metrics.py
    └── infrastructure/
        ├── hardware.py              # lscpu y decide_mpi_launch()
        ├── llm/                     # client.py, code_extractor.py, models.py
        ├── compilation/             # Dispatch por perfil: serial, openmp, mpi, avx, kokkos, go, ghighway, vectorize
        └── execution/               # process.py, output_parser.py, benchmark.py
```

---

## 🔄 Flujo de un pipeline con Opt\*

Ejemplo con `OptSerial-AVX-OptAVX-OpenMP-OptOpenMP-MPI-OptMPI` en una reserva Slurm de dos nodos de dos sockets:

```
fichero.cpp  (serial)
    │
    ├─ baseline: g++ -O3, 1 calentamiento + mediana de 11
    │
    ▼
 OptSerial [MoL]     g++ -O3. Si no mejora ≥ 3 %, se conserva el serial (n.d.)
    ▼
 AVX                 g++ -O3 -march=native
    ▼
 OptAVX [MoL]        mismos flags
    ▼
 OpenMP              g++ -O3 -fopenmp -march=native. Mide A y B
    ▼
 OptOpenMP [MoL]
    ▼
 MPI                 mpicxx -O3 -fopenmp -march=native
                     mpirun -np <nodos×sockets> --map-by ppr:1:socket:PE=<cores>
    ▼
 OptMPI [MoL]        mismo lanzamiento
    ▼
 fichero_optserial_avx_optavx_openmp_optopenmp_mpi_optmpi.cpp
```

En `OptSerial-MPI-OptMPI-OpenMP-OptOpenMP-AVX-OptAVX` el orden cambia, pero a partir de la etapa MPI la compilación y el `mpirun` siguen el perfil acumulado, no el nombre suelto de la etapa. OptSerial se compila siempre como serial, aunque el backend de prompts que toma prestado sea el de la primera etapa de traducción.

---

## 📜 Archivos de salida

En modo backend único el traducido se escribe junto al original:

| Entrada | Backend | Salida |
|---------|---------|--------|
| `hello.c` | OpenMP | `hello_openmp.c` |
| `hello.c` | MPI | `hello_mpi.c` |
| `hello.c` | AVX | `hello_avx.c` |
| `hello.c` | Vectorize | `hello_vectorize.c` |
| `hello.c` | Kokkos | `hello_kokkos.cpp` |
| `hello.c` | Go | `hello_go.go` |
| `hello.c` | GHighway | `hello_ghighway.cpp` |

En modo pipeline cada etapa deja su artefacto al lado del original. La extensión del fuente se conserva (`.cpp` sigue siendo `.cpp`):

```
hello_optserial.cpp
hello_optserial_avx.cpp
hello_optserial_avx_optavx.cpp
...
hello_optserial_avx_optavx_openmp_optopenmp_mpi_optmpi.cpp
```

Las variantes MoL (`*_mol_vN.cpp`) se borran al terminar la etapa. Con `--debug` se conservan.
