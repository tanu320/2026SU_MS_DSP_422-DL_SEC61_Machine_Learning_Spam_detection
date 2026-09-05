# Chapter 1: Introduction & Problem Statement

## 1.1 The Evolving Landscape of Social Engineering
The proliferation of automated and human-operated phone scams has precipitated a massive global crisis in cybersecurity and consumer protection. According to recent reports by the Federal Trade Commission (FTC) and global financial watchdogs, fraudulent phone calls—ranging from sophisticated technical support scams to IRS impersonations and aggressive financial manipulation—cost consumers billions of dollars annually. Unlike traditional written phishing attacks (e.g., spam emails or malicious SMS text messages), which can be effectively filtered by historical metadata or static text classification systems, voice-based social engineering attacks are highly dynamic, conversational, and occur in real-time. 

Scammers leverage complex, multi-stage narratives designed to bypass human critical thinking by inducing panic (e.g., "Your bank account has been compromised") or establishing false authority. The instantaneous nature of a phone call forces the victim into a high-pressure scenario, significantly increasing the probability of a successful attack. Consequently, defending against these attacks requires real-time, context-aware technological intervention capable of analyzing the linguistic payload of the conversation as it unfolds.

## 1.2 The Failure of Traditional Detection Mechanisms
Historically, the telecommunications industry has relied heavily on metadata analysis and heuristic-based filtering to detect fraudulent activity. Techniques include:
*   **Caller ID Blacklisting:** Maintaining databases of known malicious phone numbers.
*   **VoIP and Traffic Pattern Analysis:** Flagging high-volume, automated call centers originating from foreign IP blocks.
*   **STIR/SHAKEN Frameworks:** Cryptographically signing calls to verify the authenticity of the caller ID.

While these methods provide a foundational layer of security, they have been largely circumvented by the widespread adoption of caller ID spoofing and decentralized VoIP routing. Scammers easily hijack legitimate local numbers, completely nullifying the efficacy of blacklists. Therefore, modern scam detection cannot rely solely on *who* is calling; it must evaluate *what* is being said.

## 1.3 Problem Statement
The central challenge addressed by this research is the detection of malicious intent within a live, continuous audio stream. This presents a multi-faceted engineering problem:
1.  **Transitory Data:** Spoken language is unstructured, noisy, and prone to phonetic ambiguity. Capturing and digitizing this data in real-time requires a highly accurate Automatic Speech Recognition (ASR) system.
2.  **Long-Form Context:** Scammers are methodical. A fraudulent call may last upwards of 40 minutes, with the first 30 minutes consisting of benign trust-building dialogue. The malicious payload (e.g., "Please read me the gift card number") may be buried deep within the interaction. Traditional Natural Language Processing (NLP) models, such as early iterations of BERT, possess a strict 512-token context window (approximately 2.5 minutes of speech), making them fundamentally incapable of analyzing the overarching intent of a lengthy conversation.
3.  **Hardware Constraints:** To guarantee user privacy and prevent latency bottlenecks, the ideal inference pipeline must operate directly on edge devices (e.g., smartphones, local network appliances) or on cost-effective, shared-CPU cloud environments. Deploying state-of-the-art transformer models typically requires massive VRAM reserves on expensive NVIDIA GPUs, making widespread consumer deployment financially prohibitive.

## 1.4 Objectives and Scope
This project aims to engineer an end-to-end, edge-deployable Machine Learning pipeline capable of real-time voice scam detection. The specific objectives include:

*   **Dataset Synthesis & Governance:** Construct a balanced, robust dataset of malicious and legitimate transcripts, ensuring mathematically guaranteed zero data leakage between training and evaluation partitions to prevent artificial inflation of accuracy metrics.
*   **Cascaded Architecture Design:** Implement a multi-stage architecture utilizing a lightweight ASR model (OpenAI's Whisper) to generate transcripts, feeding into a long-context transformer (ModernBERT) capable of ingesting 8,192 tokens per forward pass.
*   **Parameter-Efficient Domain Adaptation:** Fine-tune the massive language model using Low-Rank Adaptation (LoRA) to learn the linguistic topology of scams. Subsequently, perform Domain Adaptation on noisy ASR transcripts to build robustness against phonetic spelling errors and missing punctuation.
*   **Edge Optimization and Quantization:** Sever the dependency on expensive GPU clusters by compressing the PyTorch pipeline into the GGUF binary format. Utilize low-level C++ memory mapping (`mmap`) and CPU vector instructions to achieve ultra-low latency inference on standard consumer processors.
*   **Ablation Study of Static Post-Training Quantization (PTQ):** Investigate the mathematical feasibility of compressing ModernBERT to raw INT8 precision using KL-Divergence calibration, analyzing the destructive impact of activation outliers on model precision.

## 1.5 Document Structure
The remainder of this report is structured as follows: Chapter 2 provides a comprehensive literature review of ASR evolution, transformer context windows, and model quantization. Chapter 3 details the data engineering, governance, and exploratory data analysis. Chapter 4 outlines the system architecture and the two-phase training methodology. Chapter 5 explores the optimization mechanisms for CPU deployment. Finally, Chapter 6 presents the evaluation results and proposes future directions for compact, privacy-preserving device deployment.
# Chapter 2: Literature Review & Theoretical Background

## 2.1 The Evolution of Automatic Speech Recognition (ASR)
The transcription of human speech into text has been a fundamental challenge in computer science for decades. Early ASR systems relied heavily on Hidden Markov Models (HMMs) combined with Gaussian Mixture Models (GMMs). While effective for highly structured, studio-quality recordings with limited vocabularies, these statistical models proved brittle when exposed to background noise, diverse accents, or conversational filler.

The advent of Deep Neural Networks (DNNs) revolutionized ASR. Architectures transitioned from Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks to fully attention-based transformer models. The watershed moment for open-source ASR was the release of OpenAI’s Whisper architecture in 2022. 

Whisper is an encoder-decoder transformer trained on 680,000 hours of weakly supervised, multilingual audio data. Unlike traditional ASR models that require perfect, hand-annotated phoneme transcripts, Whisper was trained to predict raw text directly from audio spectrograms. This massive scale endowed Whisper with unprecedented robustness to background noise, technical jargon, and heavily accented speech. For this project, the `whisper-tiny` variant (39M parameters) was selected. It provides a crucial balance between transcription accuracy and the ability to run inferences in real-time on consumer-grade central processing units (CPUs), avoiding the latency inherent in round-trip API calls to cloud GPU servers.

## 2.2 The Transformer Revolution and Context Windows
In 2017, Vaswani et al. introduced the Transformer architecture ("Attention Is All You Need"), fundamentally altering NLP. By utilizing self-attention mechanisms, transformers can evaluate the relationships between all words in a sequence simultaneously, rather than processing them sequentially like older RNNs. 

This led to the creation of BERT (Bidirectional Encoder Representations from Transformers) by Google in 2018. BERT models analyze text bidirectionally, gaining a deep understanding of semantic context. However, standard BERT architectures (and their lightweight derivatives like DistilBERT) are bottlenecked by a fundamental limitation: the **$O(N^2)$ quadratic complexity** of the self-attention mechanism. As the length of the input text ($N$) increases, the memory required to calculate attention scales quadratically. 

Consequently, traditional BERT models enforce a strict hard-cap of **512 tokens**. Given the standard NLP heuristic that 1 token equates to approximately 0.75 words, a 512-token limit truncates input at roughly 385 words, or ~2.5 minutes of continuous speaking at 150 words per minute. In the context of scam detection, this limitation is fatal. Scammers frequently employ "long-con" tactics, spending several minutes building rapport and false trust before introducing the malicious payload (e.g., requesting financial details). A model constrained to 512 tokens is functionally blind to the climax of the conversation.

### 2.2.1 The ModernBERT Architecture
To resolve the context bottleneck, this project utilizes **ModernBERT**, released in late 2024. ModernBERT redesigns the underlying attention mechanism, incorporating techniques such as FlashAttention and Rotary Positional Embeddings (RoPE). These architectural upgrades break the quadratic memory bottleneck, extending the maximum context window to an unprecedented **8,192 tokens**. This expansion allows the classifier to ingest approximately 6,150 words—or ~41 minutes of continuous spoken audio—in a single forward pass, guaranteeing that the malicious intent of a lengthy scam call is captured and analyzed.

## 2.3 Parameter-Efficient Fine-Tuning (PEFT): LoRA
Training massive language models (LLMs) requires adjusting millions or billions of parameters. Traditional full fine-tuning of a 149M-parameter model like ModernBERT necessitates significant VRAM to store the optimizer states and gradients, making it impossible to train on consumer hardware (e.g., 16GB T4 GPUs).

To overcome this, the project employs **Low-Rank Adaptation (LoRA)**. LoRA freezes the original pre-trained weights of the model and injects trainable rank-decomposition matrices into each transformer layer. 

Mathematically, for a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, the update is constrained by representing it with a low-rank decomposition:
$W = W_0 + \Delta W = W_0 + BA$
where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, and the rank $r \ll \min(d, k)$. 

By setting $r=16$, the number of trainable parameters is reduced to approximately ~2% of the base model. This approach yields three critical benefits:
1.  **Memory Efficiency:** Training can be executed comfortably on 2x T4 GPUs using gradient accumulation.
2.  **Rapid Convergence:** With fewer parameters to update, the Loss Function converges much faster (typically within 3-4 epochs).
3.  **Prevention of Catastrophic Forgetting:** Because the foundational weights ($W_0$) remain frozen, the model retains its vast pre-trained knowledge of the English language, learning only the specific delta ($BA$) required to identify scam intent.

## 2.4 Edge Inference and Model Quantization
Deploying PyTorch models to production environments typically requires heavy Python runtimes and dedicated CUDA environments. To deploy the pipeline on a shared CPU architecture such as Hugging Face Spaces, the model must be decoupled from PyTorch and heavily optimized.

### 2.4.1 llama.cpp and GGUF
`llama.cpp` is a hyper-optimized C++ inference engine designed to execute large language models on standard CPUs. It compiles the PyTorch architecture into the **GGUF** (GPT-Generated Unified Format) binary format. GGUF provides two massive advantages:
1.  **Memory Mapping (`mmap`):** Rather than loading a 500MB model entirely into RAM (which can trigger Out-of-Memory crashes on cheap servers), `mmap` leaves the model on the physical disk and pages only the exact required weight matrices into RAM in microseconds.
2.  **Hardware Vectorization:** `llama.cpp` leverages CPU-specific vector instructions (e.g., AVX2 on Intel/AMD, NEON on Apple/ARM) to execute matrix multiplications in parallel, simulating GPU-like throughput.

### 2.4.2 The Challenge of Static Quantization (PTQ)
Quantization compresses the model weights from 32-bit floating point (FP32) decimals into 8-bit integers (INT8), dramatically shrinking file size and increasing mathematical execution speed. However, **Static Post-Training Quantization (PTQ)** is notoriously difficult on advanced transformers.

Static PTQ requires a "Calibration" step, where sample data is fed through the FP32 model to discover the minimum, maximum, and distribution (Entropy/KL-Divergence) of the neural activations. The algorithm then maps the FP32 scale to the -128 to 127 integer bounds. Advanced models like ModernBERT employ complex activation functions (like GeGLU) that produce massive mathematical outliers. When a static PTQ algorithm attempts to compress these outliers, the vast majority of the subtle, crucial linguistic features are squashed into a single integer bucket, destroying the semantic meaning of the text. This report will detail an ablation study demonstrating this exact phenomenon and motivating a future shift toward smaller distilled student architectures for device-class inference.
# Chapter 3: Data Engineering & Governance

## 3.1 Data Scarcity and Synthesis
Acquiring high-quality, balanced data for voice scams presents a unique challenge in machine learning. Unlike image recognition or standard text sentiment analysis, real-world scam call recordings are heavily restricted by privacy regulations (e.g., GDPR, CCPA) and telecommunications wiretapping laws. Consequently, there are no massive, open-source repositories of verified scam audio. 

To overcome this data scarcity, this project engineered a unified dataset of approximately 31,000 records by synthesizing and aggregating multiple diverse sources:
*   **Kaggle Phishing, Enron, and SMS Datasets:** These datasets provided a massive foundational corpus of malicious versus legitimate text, allowing the model to learn baseline linguistic indicators of urgency, financial manipulation, and deception.
*   **Teeconnie Data & Synthetic LLM JSONs:** To bridge the gap between written text and conversational dialogue, the dataset was expanded using highly specific conversational scenarios generated by large language models (acting as both scammers and victims).
*   **Phase 2 ASR Transcripts:** A secondary dataset of raw audio files (e.g., the Bosu dataset) was processed through the OpenAI Whisper pipeline to generate noisy, imperfect transcripts, enabling the model to train on real-world acoustic transcription errors.

## 3.2 Data Governance and Leakage Prevention
In natural language processing, "Data Leakage" occurs when identical or near-identical text strings appear in both the training set and the testing set. If leakage occurs, the neural network simply memorizes the string during training and achieves a falsely high accuracy score during testing, rendering the evaluation metrics useless for real-world deployment.

To mathematically guarantee the integrity of the evaluation metrics, a strict Continuous Integration / Continuous Deployment (CI/CD) data pipeline was engineered (`src/data/02_build_datasets.py` and `03_validate_partitions.py`):
1.  **Deterministic Partitioning:** Upon ingestion, 20% of the unified dataset is randomly but deterministically carved out to form a **Global Hold-Out Set**. This test set is frozen and never exposed to the model during training.
2.  **String-Overlap Algorithms:** Before the training phase is allowed to begin, an automated script cross-references the raw strings in the training set against the hold-out set. If a single exact-string overlap is detected, the pipeline actively fails the build (raising an exception) and halts execution. This strict governance ensures that the model is evaluated purely on its ability to generalize to unseen scams.

## 3.3 Exploratory Data Analysis (EDA)
Prior to model training, extensive Exploratory Data Analysis (EDA) was conducted to understand the structural and behavioral characteristics of the dataset.

### 3.3.1 Class Imbalance Resolution
In real-world scenarios, scams represent a tiny fraction of total phone calls (e.g., 99% legitimate, 1% scam). However, training a neural network on this imbalanced distribution is highly destructive. During gradient descent, the model's loss function would quickly learn that it can achieve 99% accuracy by blindly guessing "Legitimate" for every single input. Consequently, the model would learn zero linguistic features about what constitutes a scam.

To resolve this, the training dataset was aggressively balanced to an exact **50/50 split** between malicious and legitimate transcripts. By exposing the network to an equal frequency of both classes, the gradients are perfectly balanced, eliminating statistical frequency bias. This forces the model to update its weights based exclusively on the actual linguistic differences (the semantics) between the texts. Once the model is deployed, the real-world imbalance is managed dynamically by tuning the output probability threshold (e.g., raising the threshold from 0.5 to 0.85 to restrict false positives).

### 3.3.2 Transcript Length Distributions
A critical insight emerged during the analysis of word count distributions. As documented in the Phase 2 Dataset Summary, the structural length of a call is highly correlated with its intent.

| Metric | Overall | Scams (Label 1) | Legits (Label 0) |
|--------|---------|-----------------|------------------|
| **Mean Word Count** | 38.7 | 60.5 | 16.8 |
| **Max Word Count** | 1389 | 1389 | 74 |

*   **Legitimate Calls:** Averaged only 17 words (roughly 5 to 10 seconds of speech). This accurately reflects real-world benign data, which often consists of quick voicemails or brief transactional statements (e.g., "Call me back when you get this").
*   **Scam Calls:** Averaged 61 words (25 to 40 seconds of speech), with extreme outliers reaching up to **1,389 words** (nearly 10 minutes of continuous speech). 

**Architectural Implications:** 
This massive disparity in length confirms a well-known behavioral tactic: scammers rely on long, highly scripted narratives to build a false persona, establish trust, and artificially generate urgency before demanding a payload. This EDA finding provided the empirical justification for abandoning standard BERT models (which truncate after 385 words) and adopting the 8,192-token architecture of ModernBERT. Without long-context capabilities, the model would go blind to the scammer's payload during an extended attack.
# Chapter 4: System Architecture & Methodology

## 4.1 Architectural Paradigms: Cascaded vs. End-to-End
In the domain of voice analysis, machine learning systems generally fall into two architectural paradigms: End-to-End (E2E) audio models and Cascaded pipelines. 

An E2E model (such as an Audio Spectrogram Transformer) ingests raw audio waveforms and outputs a direct classification (e.g., Scam/Not Scam) without generating intermediate text. While E2E models theoretically preserve acoustic nuances like tone of voice or emotion, they require hundreds of thousands of hours of annotated audio data to train effectively from scratch. Gathering this volume of specialized scam audio is computationally and logistically prohibitive.

This project employs a **Cascaded Architecture**:
1.  **Stage 1 (ASR):** Raw audio is ingested by OpenAI's Whisper model, which converts the acoustic signals into an intermediate text transcript.
2.  **Stage 2 (NLP):** The intermediate transcript is passed into a fine-tuned sequence classifier (ModernBERT) to determine malicious intent.

The cascaded approach allows the pipeline to leverage the pre-trained robustness of a state-of-the-art ASR model, focusing all available computational resources on fine-tuning the NLP classifier on the specific linguistic topologies of scams. Furthermore, a cascade architecture significantly improves system interpretability and debugging. If the system misclassifies an input, engineers can analyze the intermediate transcript to determine whether the error originated from an acoustic misunderstanding (an ASR failure) or a semantic misinterpretation (an NLP logic failure).

## 4.2 Phase 1: Universal Fine-Tuning Methodology
The core classification model is **ModernBERT**, selected for its 8,192-token context limit and highly optimized attention mechanism. The initial training phase focused on teaching the model the foundational concepts of scam topology using a massive corpus of clean, written text (~24,800 records).

### 4.2.1 Training Hyperparameters and Optimization
Training was executed on a distributed cluster utilizing two NVIDIA T4 GPUs (16GB VRAM each). The following hyperparameters were optimized for parameter-efficient convergence:

*   **Optimizer:** `AdamW` (Adam with Weight Decay). AdamW is the industry standard for transformer fine-tuning. It dynamically adapts the learning rate for each parameter while applying weight decay to prevent the model from growing excessively large weights, effectively mitigating overfitting on the 24K dataset.
*   **Learning Rate (LR):** A learning rate of `2e-4` was utilized. While traditional full fine-tuning typically requires a minute learning rate (e.g., `2e-5`) to avoid destroying pre-trained weights, the use of LoRA isolates the trainable parameters to new adapter matrices. These new matrices can safely absorb a higher learning rate, drastically accelerating convergence.
*   **Learning Rate Scheduler:** A Linear Decay with Warmup was implemented. The learning rate began at 0, scaled linearly to `2e-4` over the first 10% of the training steps (to prevent shocking the uninitialized LoRA adapters with massive gradients), and then decayed linearly to 0, allowing the loss to settle into a stable minimum.
*   **Batch Size & Gradient Accumulation:** To maximize hardware utilization without triggering Out-Of-Memory (OOM) errors, a batch size of 16 per GPU was used. Gradient Accumulation was employed to simulate an effective batch size of 32, ensuring smooth and stable gradient updates.
*   **Loss Function:** Binary Cross-Entropy (BCE) Loss was utilized. Cross-Entropy penalizes the model logarithmically based on confidence. If the model is extremely confident but incorrect, the loss explodes exponentially. This intense learning signal rapidly corrects erroneous biases early in the training cycle.
*   **Epochs:** The model converged rapidly, achieving peak validation accuracy within 4 epochs. Training beyond this point risked catastrophic overfitting (where the model memorizes exact training phrases rather than generalizing semantic patterns).

## 4.3 Phase 2: Domain Adaptation for ASR Noise
While Phase 1 successfully taught the model to identify scams in clean, written text (e.g., emails or SMS), real-world ASR transcripts are fundamentally different. ASR models generate output completely devoid of structural formatting; there is often no punctuation, no capitalization, and a high frequency of phonetic spelling errors (e.g., transcribing "bank account" as "banc acc").

If the Phase 1 model were directly deployed against Whisper transcripts, its accuracy would plummet when encountering these structural anomalies. Therefore, a secondary training phase—**Domain Adaptation**—was executed.

During Phase 2, the LoRA adapters from the Universal model were frozen, and a secondary layer of fine-tuning was applied using a small dataset (~2,788 records) of noisy, Whisper-generated transcripts. Because the model had already learned the core definition of a scam in Phase 1, Phase 2 required only a fraction of the training duration (approximately 140 training steps). This brief exposure was perfectly calculated: it was long enough to teach the model to ignore phonetic noise and grammatical chaos, but short enough to prevent it from overfitting to the specific transcription errors of that small dataset.
# Chapter 5: Optimization & Deployment

## 5.1 Deployment Constraints and Cloud Infrastructure
A core objective of this project was to democratize access to the scam detection pipeline. While the model was trained on high-performance GPUs, deploying an application that requires dedicated NVIDIA inference clusters is cost-prohibitive for large-scale consumer rollouts. The target deployment environment was Hugging Face Spaces—a highly accessible platform that offers a free-tier hosting architecture. However, this environment is strictly constrained to shared CPU resources, possessing no hardware acceleration and limited RAM. 

Deploying a native PyTorch model (which intrinsically attempts to parallelize tensor operations over CUDA cores) onto a shared CPU results in catastrophic latency, rendering real-time voice analysis impossible. Therefore, aggressive pipeline optimization was required.

## 5.2 The GGUF Compilation Process via llama.cpp
To sever the dependency on PyTorch and GPU hardware, the final Domain-Adapted ModernBERT model was compiled using `llama.cpp`. This open-source C++ inference engine specializes in executing Large Language Models natively on central processing units. 

The optimization pipeline involves several critical translations:
1.  **Format Conversion:** The model's weights and configuration schemas are extracted from their original `.safetensors` formatting and repacked into a singular **GGUF (GPT-Generated Unified Format)** binary file. This monolithic structure drastically improves disk read speeds during initialization.
2.  **Memory Mapping (`mmap`):** Standard PyTorch pipelines utilize RAM loading, wherein the entire 500MB model must be pushed into memory before inference can begin. The GGUF compilation inherently supports memory mapping. The host operating system treats the binary file on the solid-state drive as an extension of physical RAM, paging in specific weight matrices instantly on an as-needed basis. This eliminates startup latency and prevents Out-Of-Memory (OOM) failures on constrained cloud servers.
3.  **Hardware Vectorization:** `llama.cpp` dynamically detects the architecture of the host CPU (e.g., AVX2 or AVX-512 instruction sets for Intel/AMD, or NEON for ARM processors). It then rewrites the matrix multiplications at the core of the neural network to execute via Single Instruction, Multiple Data (SIMD) vector processing. This forces the CPU to calculate multiple weights in a single clock cycle, simulating the parallel processing topology of a GPU.

## 5.3 Quantization Execution and Latency Benchmarks
Quantization is the mathematical process of down-casting high-precision numbers into lower-precision formats. The original ModernBERT model computes using 32-bit floating-point (FP32) values. Through `llama.cpp`, these weights can be quantized into 8-bit integers (INT8) or 4-bit configurations (Q4_K_M). 

The Hugging Face CPU environment excels at integer mathematics. By shifting the computational burden from floating-point decimals to integers, the inference engine achieved a staggering performance increase. The fully quantized GGUF pipeline successfully processed incoming data and generated classifications with a total end-to-end latency of just **1.98 seconds**.

## 5.4 Ablation Study: The Failure of Static PTQ on ONNX
While `llama.cpp` achieved extraordinary success using dynamic evaluation strategies, the project also explored an alternative optimization route: **Static Post-Training Quantization (PTQ)** via Microsoft's ONNX Runtime. This pathway was investigated as a model-compression ablation and portability study.

Static PTQ requires a "Calibration Dataset." A script feeds hundreds of real transcripts through the original PyTorch model to monitor the exact numerical range of the activations passing through every layer. Using a mathematical algorithm—specifically KL-Divergence (Entropy)—the system attempts to map the ideal dynamic range of those floating-point activations down to an absolute -128 to 127 integer scale.

**The Finding:** 
When the resulting `modernbert_base_static_int8.onnx` model was evaluated against the Global Hold-Out Set, accuracy collapsed to **57.50%** (an F1 Score of 53.95%—barely above random chance). 

This ablation study provided a profound engineering insight: **Modern transformer architectures containing complex activation functions (like GeGLU) and Rotary Positional Embeddings (RoPE) produce severe mathematical outliers.** When the Static PTQ algorithm observes an outlier of `+100.0`, it forcefully expands the INT8 scale to accommodate it. This effectively squashes the standard linguistic feature representations (which typically hover between `-1.0` and `+1.0`) into a single integer bin (e.g., `0`). The resulting quantization noise completely obliterates the semantic meaning of the text. 

This outcome proves that naively applying standard Static PTQ to a state-of-the-art transformer is an unviable strategy. The more realistic next step is architectural compression: distill the transcript-trained ModernBERT teacher into a smaller MiniLM/MobileBERT-style student, then apply deployment-runtime-specific quantization to the already compact student.
# Chapter 6: Results, Evaluation & Future Work

## 6.1 Final Model Evaluation
To ensure the empirical validity of the trained pipeline, all evaluations were executed against the frozen Global Hold-Out Set. This dataset (comprising 20% of the unified corpus, approximately 6,200 records) was deterministically isolated prior to any training and remained mathematically guaranteed to contain zero string-overlap with the training partitions.

The cascading architecture was evaluated sequentially. First, raw audio was transcribed by the Whisper ASR module. The resulting noisy text was then passed through the final GGUF-quantized ModernBERT sequence classifier, utilizing the weights tuned during the Phase 2 Domain Adaptation.

### 6.1.1 Performance Metrics
The system was evaluated using standard classification metrics: Accuracy, Precision, Recall, and the F1 Score. Due to the intentional 50/50 balancing of the training data and the subsequent thresholding strategies, the F1 Score serves as the primary metric of success, elegantly balancing the trade-off between False Positives (flagging a legitimate call as a scam) and False Negatives (missing a real scam).

**End-to-End Pipeline Performance (Spoken Audio -> Text -> Classification):**
*   **F1 Score:** 97.89%
*   **Accuracy:** 97.91%

This extraordinary performance metric decisively validates the Domain Adaptation hypothesis. Despite Whisper generating frequent phonetic and grammatical errors on complex sentences, the Phase 2 fine-tuning successfully taught the classification head to ignore structural noise and focus purely on the overarching semantic intent of social engineering. 

### 6.1.2 The Impact of Threshold Calibration
During inference on the Hugging Face Spaces deployment, the `scikit-learn` prediction head defaulted to a mathematically strict 0.5 probability threshold (`pred_idx = self.gguf_head.predict([embeds])[0]`). Because the model was trained on perfectly balanced data, this 0.5 threshold was optimal for the test set.

However, recognizing that real-world telecommunications data is massively imbalanced (e.g., legitimate calls vastly outnumber fraudulent ones), the underlying inference script was designed to support threshold calibration (e.g., utilizing `.predict_proba()` to shift the positive classification boundary to 0.85). This engineering decision ensures that enterprise deployments can manually restrict the False Positive Rate (FPR), minimizing consumer friction.

## 6.2 System Telemetry and Tracking
To maintain scientific rigor and reproducibility, every iteration of the model training and evaluation process was tracked utilizing **MLflow** integrated natively with **DagsHub**.

The CI/CD telemetry pipeline recorded:
1.  **Hyperparameters:** Tracking LoRA configurations ($r$, alpha), learning rates, batch sizes, and optimizer momentum.
2.  **Metrics:** Real-time logging of Training Loss, Validation Loss, and Epoch durations.
3.  **Artifacts:** The final confusion matrices, loss curve plots, and serialized model weights were automatically versioned and pushed to the cloud bucket, enabling instantaneous rollback capabilities.

## 6.3 Conclusions
This project fundamentally demonstrates the viability of executing highly complex, state-of-the-art NLP pipelines within constrained hardware environments for the purpose of real-time security. 

Key conclusions include:
1.  **Context is King:** The statistical realization that scams average over 60 words and can exceed 1,300 words proved that traditional 512-token limit models (DistilBERT) are insufficient for conversational security. ModernBERT’s 8,192-token capacity is mandatory for identifying slow-burn social engineering tactics.
2.  **LoRA is Sufficient for Intent:** A model does not need to learn English from scratch. Fine-tuning a mere 2% of a pre-trained foundation model via Low-Rank Adaptation is entirely sufficient to teach it the complex linguistic typologies of financial fraud.
3.  **Static Quantization is Volatile on Advanced Transformers:** The catastrophic failure of Static PTQ on ONNX Runtime (dropping accuracy to 57%) revealed the danger of applying naive integer squashing to architectures utilizing RoPE and GeGLU activations. Advanced calibration or PyTorch-level constraints are necessary.

## 6.4 Future Directions: Compact Student Models for Device Deployment
While the current Hugging Face Spaces deployment successfully demonstrates CPU-only inference, it remains a cloud-dependent solution. Transmitting continuous live audio to a cloud server raises significant privacy and bandwidth concerns for consumer applications.

The more defensible future trajectory is a compact student model: use the transcript-trained ModernBERT classifier as the teacher, train a MiniLM or MobileBERT student with a combined supervised and distillation loss, and then quantize the smaller student using the target runtime's mobile tooling. This avoids forcing aggressive INT8 quantization onto a large ModernBERT graph after the PTQ ablation showed major accuracy loss. It also creates a cleaner path toward offline inference because the architecture is small before quantization begins.
