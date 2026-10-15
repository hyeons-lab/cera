import 'dart:async';
import 'dart:typed_data';

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart'
    show CeraAudioProfile, CeraTtsVoice;
import 'package:flutter/material.dart';

import '../app_constants.dart';
import '../chat_state.dart';
import '../services/audio_recorder_service.dart';
import 'audio_waveform.dart';

/// Floating Material 3 pill composer with auto-expanding text input, image
/// attachment preview, integrated audio mode & voice persona chips, and
/// push-to-talk microphone recording.
class MessageComposer extends StatefulWidget {
  const MessageComposer({
    super.key,
    required this.controller,
    required this.isBusy,
    required this.isGenerating,
    required this.canAttachImage,
    required this.canAttachAudio,
    required this.pendingImageBytes,
    required this.pendingImageName,
    required this.onSend,
    required this.onStop,
    required this.onPickImage,
    required this.onClearImage,
    required this.onSendAudio,
    this.audioRecorder,
    this.hasAudioIn = false,
    this.hasAudioOut = false,
    this.audioChatMode = AudioChatMode.interleaved,
    this.onAudioChatModeChanged,
    this.audioProfile,
    this.selectedVoicePrompt = '',
    this.onVoiceChanged,
    this.ttsSampleTexts = const <String>[],
  });

  final TextEditingController controller;
  final bool isBusy;
  final bool isGenerating;
  final bool canAttachImage;
  final bool canAttachAudio;
  final Uint8List? pendingImageBytes;
  final String? pendingImageName;
  final VoidCallback onSend;
  final VoidCallback onStop;
  final VoidCallback onPickImage;
  final VoidCallback onClearImage;
  final void Function(List<double> pcm, int sampleRate) onSendAudio;
  final AudioRecorderService? audioRecorder;
  final bool hasAudioIn;
  final bool hasAudioOut;
  final AudioChatMode audioChatMode;
  final ValueChanged<AudioChatMode>? onAudioChatModeChanged;
  final CeraAudioProfile? audioProfile;
  final String selectedVoicePrompt;
  final ValueChanged<String?>? onVoiceChanged;
  final List<String> ttsSampleTexts;

  @override
  State<MessageComposer> createState() => _MessageComposerState();
}

class _MessageComposerState extends State<MessageComposer> {
  static const _kDefaultImageName = 'Attached image';
  static const _kPendingImageSubtitle = 'Attached to next message';
  static const _kRemoveImageTooltip = 'Remove attached image';
  static const _kAttachImageTooltip = 'Attach image';
  static const _kGeneratingHint = 'Generating response...';
  static const _kIdleHint = 'Message Cera...';
  static const _kTtsIdleHint = 'Enter text to synthesize into speech...';
  static const _kStopTooltip = 'Stop generation';
  static const _kSendTooltip = 'Send message';
  static const _kSpeakTooltip = 'Synthesize speech';
  static const _kHoldToTalkTooltip = 'Hold to record voice';
  static const _kDefaultVoiceLabel = 'Default Voice';
  static const _kDefaultVoiceCode = '__default__';
  static const _kSelectVoiceTooltip = 'Select voice persona';
  static const _kSelectAudioModeTooltip = 'Select audio mode';
  static const _kSampleChipLabel = 'Sample';
  static const _kModeVoiceChat = 'Voice Chat';
  static const _kModeSpeakTts = 'Speak (TTS)';
  static const _kModeTranscribeAsr = 'Transcribe (ASR)';
  static const _kModeTextOnly = 'Text Only';
  static const _kFallbackTtsSamples = [
    'Hello, this voice was synthesized entirely on-device with Cera.',
    'The quick brown fox jumps over the lazy dog near the riverbank.',
    'Local neural speech synthesis runs privately with zero cloud latency.',
  ];
  static const double _kCancelDragThreshold = -40.0;

  AudioRecorderService? _localRecorder;

  AudioRecorderService get _audioRecorder =>
      widget.audioRecorder ?? (_localRecorder ??= AudioRecorderService());

  bool _isStartingRecording = false;
  bool _isRecordingAudio = false;
  bool _draggedToCancel = false;
  Offset? _pointerOrigin;
  int _recordingSeconds = 0;
  int _sampleIndex = 0;
  Timer? _recordTimer;

  @override
  void didUpdateWidget(MessageComposer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.audioRecorder != oldWidget.audioRecorder &&
        widget.audioRecorder != null) {
      _localRecorder?.dispose();
      _localRecorder = null;
    }
  }

  @override
  void dispose() {
    _recordTimer?.cancel();
    if (_isRecordingAudio || _isStartingRecording) {
      _audioRecorder.cancelRecording().ignore();
    }
    _localRecorder?.dispose();
    super.dispose();
  }

  Future<void> _startRecording(Offset globalPosition) async {
    if (widget.isBusy || _isRecordingAudio || _isStartingRecording) return;
    _isStartingRecording = true;
    _pointerOrigin = globalPosition;
    try {
      await _audioRecorder.startRecording(
        sampleRate: ChatDefaults.recordingSampleRate,
      );
      if (!mounted) {
        await _audioRecorder.cancelRecording();
        return;
      }
      if (!_isStartingRecording) {
        await _audioRecorder.cancelRecording();
        return;
      }
      _isStartingRecording = false;
      setState(() {
        _isRecordingAudio = true;
        _draggedToCancel = false;
        _recordingSeconds = 0;
      });
      _recordTimer?.cancel();
      _recordTimer = Timer.periodic(const Duration(seconds: 1), (_) {
        if (mounted && _isRecordingAudio) {
          setState(() => _recordingSeconds++);
        }
      });
    } catch (err) {
      _isStartingRecording = false;
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Microphone error: $err')));
      }
    }
  }

  void _onPointerMove(Offset currentPosition) {
    final origin = _pointerOrigin;
    if (!_isRecordingAudio || origin == null) return;
    final diff = currentPosition - origin;
    final isCancel =
        diff.dx < _kCancelDragThreshold || diff.dy < _kCancelDragThreshold;
    if (isCancel != _draggedToCancel) {
      setState(() => _draggedToCancel = isCancel);
    }
  }

  Future<void> _stopRecording() async {
    if (_isStartingRecording) {
      _isStartingRecording = false;
      _pointerOrigin = null;
      try {
        await _audioRecorder.cancelRecording();
      } catch (_) {}
      return;
    }
    if (!_isRecordingAudio) return;
    _recordTimer?.cancel();
    final cancelled = _draggedToCancel;
    _pointerOrigin = null;

    setState(() {
      _isRecordingAudio = false;
      _draggedToCancel = false;
    });

    if (cancelled) {
      try {
        await _audioRecorder.cancelRecording();
      } catch (_) {}
      return;
    }

    try {
      final pcm = await _audioRecorder.stopRecording();
      if (pcm.isNotEmpty && mounted) {
        widget.onSendAudio(pcm, ChatDefaults.recordingSampleRate);
      }
    } catch (err) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Recording error: $err')));
      }
    }
  }

  bool get _isTtsMode =>
      widget.hasAudioOut && widget.audioChatMode == AudioChatMode.textToSpeech;

  String _labelForAudioMode(AudioChatMode mode) {
    switch (mode) {
      case AudioChatMode.interleaved:
        return _kModeVoiceChat;
      case AudioChatMode.textToSpeech:
        return _kModeSpeakTts;
      case AudioChatMode.speechToText:
        return _kModeTranscribeAsr;
      case AudioChatMode.textOnly:
        return _kModeTextOnly;
    }
  }

  IconData _iconForAudioMode(AudioChatMode mode) {
    switch (mode) {
      case AudioChatMode.interleaved:
        return Icons.graphic_eq_rounded;
      case AudioChatMode.textToSpeech:
        return Icons.volume_up_rounded;
      case AudioChatMode.speechToText:
        return Icons.transcribe_rounded;
      case AudioChatMode.textOnly:
        return Icons.chat_bubble_outline_rounded;
    }
  }

  void _fillNextSampleScript() {
    final samples = widget.ttsSampleTexts.isNotEmpty
        ? widget.ttsSampleTexts
        : _kFallbackTtsSamples;
    final sample = samples[_sampleIndex % samples.length];
    _sampleIndex++;
    widget.controller.text = sample;
    widget.controller.selection = TextSelection.collapsed(
      offset: sample.length,
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final pendingBytes = widget.pendingImageBytes;
    final showAudioControls =
        (widget.hasAudioIn || widget.hasAudioOut) &&
        widget.onAudioChatModeChanged != null;
    final voices = widget.audioProfile?.voices ?? const <CeraTtsVoice>[];
    final showVoiceSelector =
        voices.isNotEmpty && widget.onVoiceChanged != null;
    final activeVoice = voices
        .where((v) => v.prompt == widget.selectedVoicePrompt)
        .firstOrNull;
    final hasLeftTools =
        widget.canAttachImage || showAudioControls || showVoiceSelector;

    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 14),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 768),
            child: Container(
              decoration: BoxDecoration(
                color: theme.colorScheme.surfaceContainerLow,
                borderRadius: BorderRadius.circular(26),
              ),
              padding: const EdgeInsets.fromLTRB(14, 10, 12, 10),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  if (pendingBytes != null) ...[
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Row(
                        children: [
                          ClipRRect(
                            borderRadius: BorderRadius.circular(10),
                            child: Image.memory(
                              pendingBytes,
                              width: 44,
                              height: 44,
                              fit: BoxFit.cover,
                            ),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  widget.pendingImageName ?? _kDefaultImageName,
                                  style: TextStyle(
                                    fontSize: 13,
                                    fontWeight: FontWeight.w600,
                                    color: theme.colorScheme.onSurface,
                                  ),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                ),
                                Text(
                                  _kPendingImageSubtitle,
                                  style: TextStyle(
                                    fontSize: 11.5,
                                    color: theme.colorScheme.onSurfaceVariant,
                                  ),
                                ),
                              ],
                            ),
                          ),
                          IconButton(
                            icon: Icon(
                              Icons.close_rounded,
                              size: 18,
                              color: theme.colorScheme.onSurfaceVariant,
                            ),
                            tooltip: _kRemoveImageTooltip,
                            onPressed: widget.isBusy
                                ? null
                                : widget.onClearImage,
                          ),
                        ],
                      ),
                    ),
                  ],
                  if (_isRecordingAudio)
                    Row(
                      children: [
                        Expanded(child: _buildRecordingIndicator(theme)),
                        const SizedBox(width: 8),
                        _buildMicButton(theme),
                      ],
                    )
                  else ...[
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        if (!hasLeftTools && widget.canAttachAudio) ...[
                          _buildMicButton(theme),
                          const SizedBox(width: 8),
                        ],
                        Expanded(
                          child: TextField(
                            controller: widget.controller,
                            minLines: 1,
                            maxLines: 5,
                            textInputAction: TextInputAction.send,
                            style: TextStyle(
                              fontSize: 15,
                              height: 1.4,
                              color: theme.colorScheme.onSurface,
                            ),
                            decoration: InputDecoration(
                              hintText: widget.isGenerating
                                  ? _kGeneratingHint
                                  : (_isTtsMode ? _kTtsIdleHint : _kIdleHint),
                              hintStyle: TextStyle(
                                fontSize: 15,
                                color: theme.colorScheme.onSurfaceVariant,
                              ),
                              isDense: true,
                              contentPadding: const EdgeInsets.symmetric(
                                horizontal: 4,
                                vertical: 8,
                              ),
                              border: InputBorder.none,
                              enabledBorder: InputBorder.none,
                              focusedBorder: InputBorder.none,
                            ),
                            readOnly: widget.isGenerating,
                            onSubmitted: (_) => widget.onSend(),
                          ),
                        ),
                        if (!hasLeftTools) ...[
                          const SizedBox(width: 8),
                          _buildPrimaryActionButton(theme),
                        ],
                      ],
                    ),
                    if (hasLeftTools) ...[
                      const SizedBox(height: 6),
                      Row(
                        children: [
                          Expanded(
                            child: SingleChildScrollView(
                              scrollDirection: Axis.horizontal,
                              child: Row(
                                children: [
                                  if (widget.canAttachImage) ...[
                                    IconButton(
                                      icon: Icon(
                                        Icons.add_photo_alternate_outlined,
                                        size: 20,
                                        color:
                                            theme.colorScheme.onSurfaceVariant,
                                      ),
                                      style: IconButton.styleFrom(
                                        backgroundColor: theme
                                            .colorScheme
                                            .surfaceContainerHigh,
                                        minimumSize: const Size(34, 34),
                                        padding: const EdgeInsets.all(7),
                                      ),
                                      tooltip: _kAttachImageTooltip,
                                      onPressed: widget.isBusy
                                          ? null
                                          : widget.onPickImage,
                                    ),
                                    const SizedBox(width: 6),
                                  ],
                                  if (showAudioControls) ...[
                                    _buildAudioModeSelector(theme),
                                    const SizedBox(width: 6),
                                  ],
                                  if (showVoiceSelector) ...[
                                    _buildVoiceSelector(
                                      theme,
                                      voices,
                                      activeVoice,
                                    ),
                                    const SizedBox(width: 6),
                                  ],
                                  if (_isTtsMode)
                                    _ComposerChipButton(
                                      icon: Icons.auto_fix_high_rounded,
                                      label: _kSampleChipLabel,
                                      onTap: widget.isBusy
                                          ? null
                                          : _fillNextSampleScript,
                                    ),
                                ],
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          if (widget.canAttachAudio) ...[
                            _buildMicButton(theme),
                            const SizedBox(width: 6),
                          ],
                          _buildPrimaryActionButton(theme),
                        ],
                      ),
                    ],
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildAudioModeSelector(ThemeData theme) {
    return PopupMenuButton<AudioChatMode>(
      tooltip: _kSelectAudioModeTooltip,
      color: theme.colorScheme.surface,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      onSelected: widget.onAudioChatModeChanged,
      itemBuilder: (context) => [
        if (widget.hasAudioIn && widget.hasAudioOut)
          const PopupMenuItem(
            value: AudioChatMode.interleaved,
            child: Text(_kModeVoiceChat),
          ),
        if (widget.hasAudioOut)
          const PopupMenuItem(
            value: AudioChatMode.textToSpeech,
            child: Text(_kModeSpeakTts),
          ),
        if (widget.hasAudioIn)
          const PopupMenuItem(
            value: AudioChatMode.speechToText,
            child: Text(_kModeTranscribeAsr),
          ),
        const PopupMenuItem(
          value: AudioChatMode.textOnly,
          child: Text(_kModeTextOnly),
        ),
      ],
      child: _ComposerChipPill(
        icon: _iconForAudioMode(widget.audioChatMode),
        label: _labelForAudioMode(widget.audioChatMode),
        showChevron: true,
      ),
    );
  }

  Widget _buildVoiceSelector(
    ThemeData theme,
    List<CeraTtsVoice> voices,
    CeraTtsVoice? activeVoice,
  ) {
    return PopupMenuButton<String>(
      tooltip: _kSelectVoiceTooltip,
      color: theme.colorScheme.surface,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      onSelected: (val) {
        widget.onVoiceChanged?.call(val == _kDefaultVoiceCode ? null : val);
        if (widget.hasAudioOut &&
            widget.audioChatMode != AudioChatMode.interleaved &&
            widget.audioChatMode != AudioChatMode.textToSpeech) {
          widget.onAudioChatModeChanged?.call(
            widget.hasAudioIn
                ? AudioChatMode.interleaved
                : AudioChatMode.textToSpeech,
          );
        }
      },
      itemBuilder: (context) => [
        const PopupMenuItem(
          value: _kDefaultVoiceCode,
          child: Text(_kDefaultVoiceLabel),
        ),
        for (final v in voices)
          PopupMenuItem(value: v.prompt, child: Text(v.label)),
      ],
      child: _ComposerChipPill(
        icon: Icons.record_voice_over_outlined,
        label: activeVoice?.label ?? _kDefaultVoiceLabel,
        showChevron: true,
      ),
    );
  }

  Widget _buildMicButton(ThemeData theme) {
    return Tooltip(
      message: _kHoldToTalkTooltip,
      child: Listener(
        onPointerDown: (event) => _startRecording(event.position),
        onPointerMove: (event) => _onPointerMove(event.position),
        onPointerUp: (_) => _stopRecording(),
        onPointerCancel: (_) => _stopRecording(),
        child: Container(
          width: 36,
          height: 36,
          decoration: BoxDecoration(
            color: _isRecordingAudio
                ? (_draggedToCancel
                      ? theme.colorScheme.error
                      : theme.colorScheme.primary)
                : theme.colorScheme.surfaceContainerHigh,
            shape: BoxShape.circle,
          ),
          child: Icon(
            _isRecordingAudio ? Icons.mic_rounded : Icons.mic_none_rounded,
            color: _isRecordingAudio
                ? theme.colorScheme.onPrimary
                : theme.colorScheme.onSurface,
            size: 19,
          ),
        ),
      ),
    );
  }

  Widget _buildPrimaryActionButton(ThemeData theme) {
    if (widget.isGenerating) {
      return IconButton.filled(
        style: IconButton.styleFrom(
          backgroundColor: theme.colorScheme.onSurface,
          foregroundColor: theme.colorScheme.surface,
          minimumSize: const Size(36, 36),
          padding: EdgeInsets.zero,
        ),
        icon: const Icon(Icons.stop_rounded, size: 18),
        tooltip: _kStopTooltip,
        onPressed: widget.onStop,
      );
    }

    return IconButton.filled(
      style: IconButton.styleFrom(
        backgroundColor: theme.colorScheme.primary,
        foregroundColor: theme.colorScheme.onPrimary,
        minimumSize: const Size(36, 36),
        padding: EdgeInsets.zero,
      ),
      icon: Icon(
        _isTtsMode ? Icons.volume_up_rounded : Icons.arrow_upward_rounded,
        size: 19,
      ),
      tooltip: _isTtsMode ? _kSpeakTooltip : _kSendTooltip,
      onPressed: widget.isBusy ? null : widget.onSend,
    );
  }

  Widget _buildRecordingIndicator(ThemeData theme) {
    final color = _draggedToCancel
        ? theme.colorScheme.error
        : theme.colorScheme.primary;

    return RecordingWaveformIndicator(
      recordingSeconds: _recordingSeconds,
      isCancelled: _draggedToCancel,
      color: color,
    );
  }
}

class _ComposerChipPill extends StatelessWidget {
  const _ComposerChipPill({
    required this.icon,
    required this.label,
    this.showChevron = false,
  });

  final IconData icon;
  final String label;
  final bool showChevron;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      height: 32,
      padding: const EdgeInsets.symmetric(horizontal: 10),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHigh,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: theme.colorScheme.onSurfaceVariant),
          const SizedBox(width: 5),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w500,
              color: theme.colorScheme.onSurface,
            ),
          ),
          if (showChevron) ...[
            const SizedBox(width: 2),
            Icon(
              Icons.keyboard_arrow_down_rounded,
              size: 15,
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ],
        ],
      ),
    );
  }
}

class _ComposerChipButton extends StatelessWidget {
  const _ComposerChipButton({
    required this.icon,
    required this.label,
    this.onTap,
  });

  final IconData icon;
  final String label;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Material(
      color: theme.colorScheme.surfaceContainerHigh,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          height: 32,
          padding: const EdgeInsets.symmetric(horizontal: 10),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 14, color: theme.colorScheme.onSurfaceVariant),
              const SizedBox(width: 5),
              Text(
                label,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                  color: theme.colorScheme.onSurface,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
