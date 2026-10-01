import 'package:cera_ffi_flutter/cera_ffi_flutter.dart' hide ModelSource;
import 'package:flutter/material.dart';

/// The Voice Persona setting: a dropdown over the voices the loaded model's
/// [profile] lists, or nothing when it lists none.
///
/// A voice the model was not trained on makes it answer in text and never
/// speak, so [saved] is shown only when this model offers it; any other saved
/// choice (one made under a different model) reads as the default voice.
/// [onChanged] gets the chosen voice's `prompt`, or null for the default.
class VoicePersonaPicker extends StatelessWidget {
  const VoicePersonaPicker({
    super.key,
    required this.profile,
    required this.saved,
    required this.onChanged,
  });

  final CeraAudioProfile? profile;
  final String saved;
  final ValueChanged<String?> onChanged;

  @override
  Widget build(BuildContext context) {
    final voices = profile?.voices ?? const <CeraTtsVoice>[];
    if (voices.isEmpty) return const SizedBox.shrink();
    final theme = Theme.of(context);
    return ListTile(
      contentPadding: EdgeInsets.zero,
      title: const Text('Voice Persona'),
      subtitle: Text(
        'Select speaker timbre and accent for synthesized speech responses.',
        style: TextStyle(
          fontSize: 12,
          color: theme.colorScheme.onSurfaceVariant,
        ),
      ),
      trailing: DropdownButton<String?>(
        value: voices.any((v) => v.prompt == saved) ? saved : null,
        dropdownColor: theme.colorScheme.surface,
        underline: const SizedBox.shrink(),
        items: [
          const DropdownMenuItem(value: null, child: Text('Default Voice')),
          for (final voice in voices)
            DropdownMenuItem(value: voice.prompt, child: Text(voice.label)),
        ],
        onChanged: onChanged,
      ),
    );
  }
}
