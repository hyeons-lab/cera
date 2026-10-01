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

  test('the audio mode follows which prompt it is, not the words in it', () {
    // A manifest may phrase its prompts any way: only the field matters.
    const custom = CeraAudioProfile(
      ttsSystemPrompt: 'Speak.',
      interleavedSystemPrompt: 'Talk and write.',
      voices: [
        CeraTtsVoice(
          label: 'Ana',
          prompt: 'Voice: ana.',
          ttsSystemPrompt: 'Speak. Voice: ana.',
          interleavedSystemPrompt: 'Talk and write. Voice: ana.',
        ),
      ],
    );
    expect(custom.audioModeOf('Speak.'), CeraAudioMode.sequential);
    expect(
      custom.audioModeOf(' Speak. Voice: ana. '),
      CeraAudioMode.sequential,
    );
    expect(custom.audioModeOf('Talk and write.'), CeraAudioMode.interleaved);
    expect(
      custom.audioModeOf('Talk and write. Voice: ana.'),
      CeraAudioMode.interleaved,
    );
  });

  test('a caller-written prompt is read the way the models read it', () {
    const profile = CeraAudioProfile.plain();
    expect(
      profile.audioModeOf('Respond with interleaved text and audio. Be brief.'),
      CeraAudioMode.interleaved,
    );
    expect(
      profile.audioModeOf('Perform TTS in french.'),
      CeraAudioMode.sequential,
    );
    expect(profile.audioModeOf('Respond to the user.'), CeraAudioMode.textOnly);
    expect(profile.audioModeOf('Perform ASR.'), CeraAudioMode.textOnly);
  });

  test('when both prompts are the same text the interleaved mode wins', () {
    const same = CeraAudioProfile(
      ttsSystemPrompt: 'Speak.',
      interleavedSystemPrompt: 'Speak.',
    );
    expect(same.audioModeOf('Speak.'), CeraAudioMode.interleaved);
  });

  test('the plain profile is the generic one every model accepts', () {
    const plain = CeraAudioProfile.plain();
    expect(plain.ttsSystemPrompt, 'Perform TTS.');
    expect(
      plain.interleavedSystemPrompt,
      'Respond with interleaved text and audio.',
    );
    expect(plain.hasVoices, isFalse);
  });

  test('the default system prompt follows the model\'s audio output', () {
    // The first voice's prompt on a model with voices, so it names that voice.
    expect(
      english.defaultSystemPromptFor(audioOut: true),
      'Respond with interleaved text and audio. Use the US female voice.',
    );
    expect(
      japanese.defaultSystemPromptFor(audioOut: true),
      'Respond with interleaved text and audio.',
    );
    expect(
      english.defaultSystemPromptFor(audioOut: false),
      ceraTextOnlySystemPrompt,
    );
  });
}
