/// Shared constants, preference keys, and UI copy for the Cera example app.
library;

/// SharedPreferences storage keys.
abstract final class PrefKeys {
  static const downloadedModels = 'cera_downloaded_models';
  static const backend = 'cera_backend';
  static const turboQuant = 'cera_turboquant';
  static const maxImageDim = 'cera_max_image_dim';
  static const audioChatMode = 'cera_audio_chat_mode';
  static const chatVoice = 'cera_chat_voice';
  static const legacyTtsVoice = 'cera_tts_voice';
  static const ttsStudioVoice = 'cera_tts_studio_voice';
  static const temperature = 'cera_temperature';
  static const topP = 'cera_top_p';
  static const topK = 'cera_top_k';
  static const maxTokens = 'cera_max_tokens';
  static const lastBundleName = 'cera_last_bundle_name';
  static const lastBundleQuant = 'cera_last_bundle_quant';
  static const themeMode = 'cera_theme_mode';
}

/// Serialized preference token values.
abstract final class PrefValues {
  static const backendCpu = 'cpu';
  static const backendGpu = 'gpu';
  static const backendHexagon = 'hexagon';

  static const audioAsr = 'asr';
  static const audioSpeechToText = 'speechToText';
  static const audioTts = 'tts';
  static const audioTextToSpeech = 'textToSpeech';
  static const audioTextOnly = 'textOnly';
  static const audioInterleaved = 'interleaved';

  static const themeLight = 'light';
  static const themeDark = 'dark';
  static const themeSystem = 'system';
}

/// JSON keys for [DownloadedModelRecord] persistence.
abstract final class ModelRecordKeys {
  static const bundleName = 'bundleName';
  static const quant = 'quant';
  static const displayName = 'displayName';
}

/// Conversation turn role identifiers.
abstract final class TurnRole {
  static const user = 'user';
  static const assistant = 'assistant';
}

/// File extensions, suffixes, and web runtime asset paths.
abstract final class ModelFormat {
  static const ggufSuffix = '-GGUF';
  static const ggufExtension = '.gguf';
  static const jsonExtension = '.json';
  static const workerUrl = 'cera/cera_worker.js';
  static const moduleUrl = 'cera/cera_wasm.js';

  /// Strips a trailing `-GGUF` suffix from a catalog bundle name if present.
  static String stripGgufSuffix(String bundleName) {
    return bundleName.endsWith(ggufSuffix)
        ? bundleName.substring(0, bundleName.length - ggufSuffix.length)
        : bundleName;
  }
}

/// Numeric defaults for inference and media preprocessing.
abstract final class ChatDefaults {
  static const int maxImageLongSize = 256;
  static const int hdImageLongSize = 512;
  static const int maxTokens = 512;
  static const int minMaxTokens = 16;
  static const int maxMaxTokens = 4096;
  static const int vocoderSampleRate = 24000;
  static const int recordingSampleRate = 16000;
  static const double greedyTemperature = 0.0;
  static const double balancedTemperature = 0.3;
  static const double creativeTemperature = 0.7;
  static const double maxSliderTemperature = 1.5;
  static const List<int> maxTokenChoices = [256, 512, 1024, 2048];
}

/// Chat template BOS and turn-continuation delimiters.
abstract final class ChatTemplateMarkers {
  static const List<String> bosPrefixes = [
    '<|startoftext|>',
    '<s>',
    '<|begin_of_text|>',
    '<bos>',
  ];

  static const imStartUser = '<|im_start|>user\n';
  static const imEndPrefix = '<|im_end|>\n';
  static const llama3UserHeader = '<|start_header_id|>user<|end_header_id|>';
  static const llama3EotPrefix = '<|eot_id|>';
  static const gemmaUserTurn = '<start_of_turn>user\n';
  static const gemmaEndTurnPrefix = '<end_of_turn>\n';
}

/// Shared user-facing strings and status labels.
abstract final class AppStrings {
  static const appTitle = 'Cera';
  static const appVersion = 'v0.7.0';
  static const metaSeparator = ' · ';

  // Status & lifecycle messages
  static const initialStatus =
      'Download a published model, or open a .gguf, to start.';
  static const modelUnloadedStatus = 'Model unloaded.';
  static const unknownModel = 'unknown';
  static const unknownBackend = 'Unknown Backend';
  static const defaultImagePrompt = 'Describe this image.';
  static const noSpeechDetected = '(No speech detected)';
  static const unsupportedContinuationMessage =
      'This model’s chat format does not support continuing this '
      'conversation. Start a new chat.';

  // Turn generation status indicators
  static const statusAnalyzingImage = 'Analyzing image...';
  static const statusSynthesizingSpeech = 'Synthesizing neural speech...';
  static const statusThinkingAndSpeaking = 'Thinking & speaking...';
  static const statusThinking = 'Thinking...';
  static const statusEncodingImage = 'Encoding image patches...';
  static const statusGeneratingResponse = 'Generating response...';
  static const statusTranscribingSpeech = 'Transcribing speech...';
  static const statusProcessingAudio = 'Processing audio...';

  // Capability tags
  static const tagVision = ' · Vision';
  static const tagVoice = ' · Voice';
  static const tagAsr = ' · ASR';
  static const tagAudio = ' · Audio';

  // Sampling labels
  static const samplingAuto = 'Auto';
  static const samplingGreedy = 'Greedy';
  static const samplingBalanced = 'Balanced (0.3)';
  static const samplingCreative = 'Creative (0.7)';

  // Clipboard & snackbars
  static const copiedToClipboard = 'Copied to clipboard';
  static const copyTextTooltip = 'Copy text';
}
