import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../services/audio_recorder_service.dart';

/// Bottom message composer with text input, image attachment preview,
/// vision picker trigger, audio push-to-talk microphone trigger, and send/stop button.
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

  @override
  State<MessageComposer> createState() => _MessageComposerState();
}

class _MessageComposerState extends State<MessageComposer> {
  final AudioRecorderService _audioRecorder = AudioRecorderService();
  bool _isStartingRecording = false;
  bool _isRecordingAudio = false;
  bool _draggedToCancel = false;
  Offset? _pointerOrigin;
  int _recordingSeconds = 0;
  Timer? _recordTimer;

  @override
  void dispose() {
    _recordTimer?.cancel();
    _audioRecorder.dispose();
    super.dispose();
  }

  Future<void> _startRecording(Offset globalPosition) async {
    if (widget.isBusy || _isRecordingAudio || _isStartingRecording) return;
    _isStartingRecording = true;
    _pointerOrigin = globalPosition;
    try {
      await _audioRecorder.startRecording(sampleRate: 16000);
      if (!mounted) {
        await _audioRecorder.cancelRecording();
        return;
      }
      if (!_isStartingRecording) {
        // User already released finger before recording finished starting
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
    if (!_isRecordingAudio || _pointerOrigin == null) return;
    final diff = currentPosition - _pointerOrigin!;
    // Dragging left or upwards by more than 40 pixels triggers cancel
    final isCancel = diff.dx < -40 || diff.dy < -40;
    if (isCancel != _draggedToCancel) {
      setState(() => _draggedToCancel = isCancel);
    }
  }

  Future<void> _stopRecording() async {
    if (_isStartingRecording) {
      _isStartingRecording = false;
      _pointerOrigin = null;
      await _audioRecorder.cancelRecording();
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
      await _audioRecorder.cancelRecording();
      return;
    }

    final pcm = await _audioRecorder.stopRecording();
    if (pcm.isNotEmpty && mounted) {
      widget.onSendAudio(pcm, 16000);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (widget.pendingImageBytes != null)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            color: theme.colorScheme.surface,
            child: Row(
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: Image.memory(
                    widget.pendingImageBytes!,
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
                        widget.pendingImageName ?? 'Attached image',
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: theme.colorScheme.onSurface,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      Text(
                        'Will be sent with next prompt',
                        style: TextStyle(
                          fontSize: 11,
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
                  tooltip: 'Remove attached image',
                  onPressed: widget.isBusy ? null : widget.onClearImage,
                ),
              ],
            ),
          ),
        Container(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            border: Border(
              top: BorderSide(
                color: theme.colorScheme.outlineVariant,
                width: 1,
              ),
            ),
          ),
          child: Row(
            children: [
              if (widget.canAttachImage && !_isRecordingAudio) ...[
                IconButton(
                  icon: Icon(
                    Icons.add_photo_alternate_outlined,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                  tooltip: 'Attach image for vision prompt',
                  onPressed: widget.isBusy ? null : widget.onPickImage,
                ),
                const SizedBox(width: 4),
              ],
              if (widget.canAttachAudio) ...[
                Listener(
                  onPointerDown: (event) => _startRecording(event.position),
                  onPointerMove: (event) => _onPointerMove(event.position),
                  onPointerUp: (_) => _stopRecording(),
                  onPointerCancel: (_) => _stopRecording(),
                  child: Container(
                    decoration: BoxDecoration(
                      color: _isRecordingAudio
                          ? (_draggedToCancel
                                ? theme.colorScheme.error
                                : theme.colorScheme.primary)
                          : Colors.transparent,
                      shape: BoxShape.circle,
                    ),
                    padding: const EdgeInsets.all(8),
                    child: Icon(
                      _isRecordingAudio
                          ? Icons.mic_rounded
                          : Icons.mic_none_rounded,
                      color: _isRecordingAudio
                          ? theme.colorScheme.onPrimary
                          : theme.colorScheme.primary,
                      size: 22,
                    ),
                  ),
                ),
                const SizedBox(width: 6),
              ],
              Expanded(
                child: _isRecordingAudio
                    ? _buildRecordingIndicator(theme)
                    : TextField(
                        controller: widget.controller,
                        decoration: InputDecoration(
                          hintText: widget.isGenerating
                              ? 'Model is generating response...'
                              : 'Send a message...',
                          hintStyle: TextStyle(
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                          filled: true,
                          fillColor: theme.scaffoldBackgroundColor,
                          contentPadding: const EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 12,
                          ),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(24),
                            borderSide: BorderSide(
                              color: theme.colorScheme.outline,
                            ),
                          ),
                          enabledBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(24),
                            borderSide: BorderSide(
                              color: theme.colorScheme.outline,
                            ),
                          ),
                          focusedBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(24),
                            borderSide: BorderSide(
                              color: theme.colorScheme.primary,
                            ),
                          ),
                        ),
                        enabled: !widget.isGenerating,
                        onSubmitted: (_) => widget.onSend(),
                      ),
              ),
              const SizedBox(width: 10),
              if (widget.isGenerating)
                IconButton.filled(
                  style: IconButton.styleFrom(
                    backgroundColor: theme.colorScheme.error,
                    foregroundColor: theme.colorScheme.onError,
                  ),
                  icon: const Icon(Icons.stop_rounded),
                  tooltip: 'Stop generation',
                  onPressed: widget.onStop,
                )
              else if (!_isRecordingAudio)
                IconButton.filled(
                  style: IconButton.styleFrom(
                    backgroundColor: theme.colorScheme.primary,
                    foregroundColor: theme.colorScheme.onPrimary,
                  ),
                  icon: const Icon(Icons.arrow_upward_rounded),
                  tooltip: 'Send message',
                  onPressed: widget.isBusy ? null : widget.onSend,
                ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildRecordingIndicator(ThemeData theme) {
    final minutes = (_recordingSeconds ~/ 60).toString().padLeft(2, '0');
    final seconds = (_recordingSeconds % 60).toString().padLeft(2, '0');

    final color = _draggedToCancel
        ? theme.colorScheme.error
        : theme.colorScheme.primary;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: color),
      ),
      child: Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 8),
          Text(
            '$minutes:$seconds',
            style: TextStyle(
              fontWeight: FontWeight.w600,
              fontSize: 12,
              color: theme.colorScheme.onSurface,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _draggedToCancel
                  ? 'Release to cancel'
                  : 'Recording... (slide away to cancel)',
              style: TextStyle(fontSize: 11, color: color),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}
