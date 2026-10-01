import 'package:cera_ffi/cera_ffi.dart';
import 'package:test/test.dart';

/// `CeraAudioProfile` only selects among strings cera already resolved: no
/// prompt is ever assembled here, so a client cannot drift from the core.
void main() {
  const female = CeraTtsVoice(
    label: 'US Female',
    prompt: 'Use the US female voice.',
    ttsSystemPrompt: 'Perform TTS. Use the US female voice.',
    interleavedSystemPrompt:
        'Respond with interleaved text and audio. Use the US female voice.',
  );
  const male = CeraTtsVoice(
    label: 'UK Male',
    prompt: 'Use the UK male voice.',
    ttsSystemPrompt: 'Perform TTS. Use the UK male voice.',
    interleavedSystemPrompt:
        'Respond with interleaved text and audio. Use the UK male voice.',
  );
  const english = CeraAudioProfile(
    ttsSystemPrompt: 'Perform TTS. Use the US female voice.',
    interleavedSystemPrompt:
        'Respond with interleaved text and audio. Use the US female voice.',
    voices: [female, male],
  );
  const japanese = CeraAudioProfile(
    ttsSystemPrompt: 'Perform TTS in japanese.',
    interleavedSystemPrompt: 'Respond with interleaved text and audio.',
  );

  test('a listed voice is used', () {
    expect(english.voiceFor('Use the UK male voice.'), same(male));
    expect(
      english.ttsSystemPromptFor(' Use the UK male voice. '),
      'Perform TTS. Use the UK male voice.',
    );
    expect(
      english.interleavedSystemPromptFor('Use the UK male voice.'),
      'Respond with interleaved text and audio. Use the UK male voice.',
    );
  });

  test(
    'an unlisted, blank or missing choice falls back to the first voice',
    () {
      for (final saved in ['Use the Martian voice.', '  ', '', null]) {
        expect(english.voiceFor(saved), same(female), reason: '$saved');
        expect(
          english.ttsSystemPromptFor(saved),
          'Perform TTS. Use the US female voice.',
          reason: '$saved',
        );
      }
    },
  );

  test('a model without voices never receives one', () {
    expect(japanese.hasVoices, isFalse);
    expect(japanese.voiceFor('Use the US female voice.'), isNull);
    // The saved default of a client built for the English model.
    expect(
      japanese.ttsSystemPromptFor('Use the US female voice.'),
      'Perform TTS in japanese.',
    );
    expect(
      japanese.interleavedSystemPromptFor('Use the US female voice.'),
      'Respond with interleaved text and audio.',
    );
  });
}
