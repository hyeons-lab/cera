import 'dart:typed_data';

import 'model_source.dart';

/// Sealed hierarchy of all user and system intents processed by [ChatController].
sealed class ChatIntent {
  const ChatIntent();
}

/// Intent to restore the last used bundle model from persistent storage.
class RestoreLastModelIntent extends ChatIntent {
  const RestoreLastModelIntent();
}

/// Intent to download and load a model bundle from the catalog.
class LoadBundleIntent extends ChatIntent {
  const LoadBundleIntent({
    required this.bundleName,
    required this.quant,
    required this.displayName,
    required this.storeDir,
  });

  final String bundleName;
  final String quant;
  final String displayName;
  final Future<String?> Function() storeDir;
}

/// Intent to load a local .gguf file model.
class LoadLocalModelIntent extends ChatIntent {
  const LoadLocalModelIntent(this.source);
  final ModelSource source;
}

/// Intent to explicitly unload the active model and release its engine resources.
class UnloadModelIntent extends ChatIntent {
  const UnloadModelIntent();
}

/// Intent to send a prompt (and optional attached image) to the model.
class SendMessageIntent extends ChatIntent {
  const SendMessageIntent(this.prompt);
  final String prompt;
}

/// Intent to send an audio prompt to an audio-in model.
class SendAudioPromptIntent extends ChatIntent {
  const SendAudioPromptIntent({
    required this.pcmSamples,
    this.sampleRate = 16000,
    this.prompt = '',
  });

  final List<double> pcmSamples;
  final int sampleRate;
  final String prompt;
}

/// Intent to cancel/stop an ongoing text generation.
class StopGenerationIntent extends ChatIntent {
  const StopGenerationIntent();
}

/// Intent to attach an image for multimodal vision prompting.
class AttachImageIntent extends ChatIntent {
  const AttachImageIntent({required this.bytes, required this.name});
  final Uint8List bytes;
  final String name;
}

/// Intent to clear the pending attached image.
class ClearAttachedImageIntent extends ChatIntent {
  const ClearAttachedImageIntent();
}

/// Intent to clear the conversation transcript.
class ClearTranscriptIntent extends ChatIntent {
  const ClearTranscriptIntent();
}
