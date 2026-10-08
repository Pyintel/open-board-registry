---
annotations_creators:
- machine-generated
language:
- en
language_creators:
- machine-generated
license: apache-2.0
multilinguality:
- monolingual
pretty_name: PyIntel Embedded Hardware CoT (The Hardware Architect)
size_categories:
- 1K<n<10K
source_datasets:
- pyintel/open-board-registry
tags:
- hardware
- embedded
- robotics
- chain-of-thought
- reasoning
- gemma
- physical-ai
task_categories:
- question-answering
- text-generation
---

# ⚡ PyIntel Embedded Hardware CoT (The Hardware Architect)

> **Zero-hallucination physical constraint reasoning grounded directly in 1,717 microcontrollers and development boards.**

`pyintel/embedded-hardware-cot` is a specialized chain-of-thought (CoT) reasoning dataset designed to teach LLMs how to solve strict physical, electrical, and computational constraints in embedded systems without hallucinating specs or recommending circuits that would fry real silicon.

Grounded in [pyintel/open-board-registry](https://huggingface.co/datasets/pyintel/open-board-registry) (comprising 1,717 verified microcontroller boards across STMicroelectronics, Espressif, Raspberry Pi, Microchip/Atmel, Nordic Semiconductor, and more), this dataset trains models to perform multi-step engineering deductions under native Gemma `<|channel>thought` tokens.

---

## 🎯 What This Dataset Teaches

Standard general-purpose LLMs routinely hallucinate microcontroller capabilities—suggesting an ATmega328P for TLS 1.3 encryption, recommending 5V logic for 3.3V I2C sensors, or exceeding SRAM limits when allocating display framebuffers. 

**PyIntel Embedded Hardware CoT** trains the model to act as a rigorous Systems Architect:

1. **Parameter Decomposition**: Extracts voltage rails, network buffers, peripheral buses, and timing deadlines from engineering queries.
2. **Mathematical Memory Sizing**: Computes exact SRAM formulas for RTOS dynamic heaps ($\text{Kernel} + \text{LwIP} + \text{mbedTLS}$), graphics framebuffers ($W \times H \times \text{BPP}$), and DMA ping-pong buffers.
3. **Negative Elimination**: Explicitly rejects incompatible board families with technical justifications (e.g. *Reject ATmega328P: 2 KB SRAM is insufficient for 128 KB heap requirement; 5V logic would destroy 3.3V sensors*).
4. **Registry Verification**: Matches requirements against verified boards in the database, evaluating clock speed, pinmuxing, and hardware peripherals.
5. **Physical Safeguards**: Provides exact pinout mappings, termination resistor values, decoupling advice, and brownout reset thresholds.

---

## 📊 Dataset Structure & Schema

Each sample contains:
- `id`: Unique record identifier (`pyintel_hw_cot_xxxxxx`).
- `domain`: One of 8 core engineering domains.
- `instruction`: Realistic developer / systems engineer requirement.
- `reasoning`: Multi-step deductive analysis formatted inside `<|channel>thought`.
- `response`: Authoritative markdown solution table with specifications and wiring notes.
- `text`: Native turn-formatted conversation ready for direct SFT fine-tuning:
  ```
  <start_of_turn>user
  {instruction}<end_of_turn>
  <start_of_turn>model
  <|channel>thought
  {reasoning}<channel|>
  {response}<end_of_turn>
  ```
- `target_board_id`: Verified ID in `open-board-registry`.
- `mcu`: Microcontroller part number.
- `vendor`: Board manufacturer / maintainer.
- `ram_bytes`: Exact SRAM capacity in bytes.
- `rom_bytes`: Exact Flash capacity in bytes.

---

## 🔬 Supported Engineering Domains

| Domain | Key Physical Constraints Evaluated |
| :--- | :--- |
| **`battery_iot`** | 3.3V logic, LiPo battery cutoff, low-quiescent LDO, TLS 1.3 socket heap sizing ($\ge 128\text{ KB}$), WiFi/BLE |
| **`automotive_industrial_can`** | Wide DC input (7V–28V DC), CAN 2.0B / CAN-FD hardware controllers, 1 kHz packet latency, 120$\Omega$ termination |
| **`robotics_foc_motion`** | Field Oriented Control (FOC), 20–40 kHz center-aligned PWM, dead-time insertion, synchronized ADC, QEI/PIO |
| **`display_framebuffer_gui`** | Framebuffer math ($W \times H \times 2$), LVGL GUI heap, high-speed SPI/QSPI DMA, refresh rates |
| **`edge_audio_dsp`** | I2S MEMS mic acquisition, CMSIS-DSP SIMD acceleration, hardware single-precision FPU, ping-pong DMA |
| **`multi_bus_sensor_fusion`** | Multi-peripheral bus contention resolution (independent SPI0, SPI1, I2C, and UART channels) |
| **`native_usb_emulation`** | On-chip native USB 2.0 Full-Speed PHY, TinyUSB stack, programmable HID/MIDI/CDC descriptors |
| **`crypto_secure_gateway`** | Hardware TRNG, AES-128/256 and SHA accelerators, secure boot ROM, Flash encryption |

---

## 🛠️ Usage with Hugging Face Datasets

```python
from datasets import load_dataset

ds = load_dataset("pyintel/embedded-hardware-cot")
print(f"Total samples: {len(ds['train']):,}")
print(ds["train"][0]["text"])
```

---

## 📜 License & Citation

Released under the **Apache 2.0 License** by the **PyIntel Research Organization**.
