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
    if (widget.isBusy || _isRecordingAudio) return;
    try {
      _pointerOrigin = globalPosition;
      await _audioRecorder.startRecording(sampleRate: 16000);
      if (!mounted) return;
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
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (widget.pendingImageBytes != null)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            color: const Color(0xFF14161B),
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
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: Color(0xFFF1F5F9),
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const Text(
                        'Will be sent with next prompt',
                        style: TextStyle(
                          fontSize: 11,
                          color: Color(0xFF8E95A5),
                        ),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  icon: const Icon(
                    Icons.close_rounded,
                    size: 18,
                    color: Color(0xFF94A3B8),
                  ),
                  tooltip: 'Remove attached image',
                  onPressed: widget.isBusy ? null : widget.onClearImage,
                ),
              ],
            ),
          ),
        Container(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
          decoration: const BoxDecoration(
            color: Color(0xFF14161B),
            border: Border(top: BorderSide(color: Color(0xFF1E222D), width: 1)),
          ),
          child: Row(
            children: [
              if (widget.canAttachImage && !_isRecordingAudio) ...[
                IconButton(
                  icon: const Icon(
                    Icons.add_photo_alternate_outlined,
                    color: Color(0xFF94A3B8),
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
                                ? const Color(0xFFEF4444)
                                : const Color(0xFF3B82F6))
                          : Colors.transparent,
                      shape: BoxShape.circle,
                    ),
                    padding: const EdgeInsets.all(8),
                    child: Icon(
                      _isRecordingAudio
                          ? Icons.mic_rounded
                          : Icons.mic_none_rounded,
                      color: _isRecordingAudio
                          ? Colors.white
                          : const Color(0xFF60A5FA),
                      size: 22,
                    ),
                  ),
                ),
                const SizedBox(width: 6),
              ],
              Expanded(
                child: _isRecordingAudio
                    ? _buildRecordingIndicator()
                    : TextField(
                        controller: widget.controller,
                        decoration: InputDecoration(
                          hintText: widget.isGenerating
                              ? 'Model is generating response...'
                              : 'Send a message...',
                          hintStyle: const TextStyle(color: Color(0xFF64748B)),
                          filled: true,
                          fillColor: const Color(0xFF0B0C0E),
                          contentPadding: const EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 12,
                          ),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(24),
                            borderSide: const BorderSide(
                              color: Color(0xFF232732),
                            ),
                          ),
                          enabledBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(24),
                            borderSide: const BorderSide(
                              color: Color(0xFF232732),
                            ),
                          ),
                          focusedBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(24),
                            borderSide: const BorderSide(
                              color: Color(0xFF3B82F6),
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
                    backgroundColor: const Color(0xFFEF4444),
                    foregroundColor: Colors.white,
                  ),
                  icon: const Icon(Icons.stop_rounded),
                  tooltip: 'Stop generation',
                  onPressed: widget.onStop,
                )
              else if (!_isRecordingAudio)
                IconButton.filled(
                  style: IconButton.styleFrom(
                    backgroundColor: const Color(0xFF3B82F6),
                    foregroundColor: Colors.white,
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

  Widget _buildRecordingIndicator() {
    final minutes = (_recordingSeconds ~/ 60).toString().padLeft(2, '0');
    final seconds = (_recordingSeconds % 60).toString().padLeft(2, '0');

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: _draggedToCancel
            ? const Color(0xFF3B1219)
            : const Color(0xFF182234),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: _draggedToCancel
              ? const Color(0xFFEF4444)
              : const Color(0xFF3B82F6),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: _draggedToCancel
                  ? const Color(0xFFEF4444)
                  : const Color(0xFF60A5FA),
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 8),
          Text(
            '$minutes:$seconds',
            style: const TextStyle(
              fontWeight: FontWeight.w600,
              fontSize: 12,
              color: Color(0xFFF1F5F9),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _draggedToCancel
                  ? 'Release to cancel'
                  : 'Recording... (slide away to cancel)',
              style: TextStyle(
                fontSize: 11,
                color: _draggedToCancel
                    ? const Color(0xFFFCA5A5)
                    : const Color(0xFF93C5FD),
              ),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}
