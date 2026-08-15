import 'dart:typed_data';
import 'package:flutter/material.dart';

/// Bottom message composer with text input, image attachment preview,
/// vision picker trigger, and send/stop button.
class MessageComposer extends StatelessWidget {
  const MessageComposer({
    super.key,
    required this.controller,
    required this.isBusy,
    required this.isGenerating,
    required this.canAttachImage,
    required this.pendingImageBytes,
    required this.pendingImageName,
    required this.onSend,
    required this.onStop,
    required this.onPickImage,
    required this.onClearImage,
  });

  final TextEditingController controller;
  final bool isBusy;
  final bool isGenerating;
  final bool canAttachImage;
  final Uint8List? pendingImageBytes;
  final String? pendingImageName;
  final VoidCallback onSend;
  final VoidCallback onStop;
  final VoidCallback onPickImage;
  final VoidCallback onClearImage;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (pendingImageBytes != null)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            color: const Color(0xFF14161B),
            child: Row(
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: Image.memory(
                    pendingImageBytes!,
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
                        pendingImageName ?? 'Attached image',
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
                  onPressed: isBusy ? null : onClearImage,
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
              if (canAttachImage) ...[
                IconButton(
                  icon: const Icon(
                    Icons.add_photo_alternate_outlined,
                    color: Color(0xFF94A3B8),
                  ),
                  tooltip: 'Attach image for vision prompt',
                  onPressed: isBusy ? null : onPickImage,
                ),
                const SizedBox(width: 4),
              ],
              Expanded(
                child: TextField(
                  controller: controller,
                  decoration: InputDecoration(
                    hintText: isGenerating
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
                      borderSide: const BorderSide(color: Color(0xFF232732)),
                    ),
                    enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(24),
                      borderSide: const BorderSide(color: Color(0xFF232732)),
                    ),
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(24),
                      borderSide: const BorderSide(color: Color(0xFF3B82F6)),
                    ),
                  ),
                  enabled: !isGenerating,
                  onSubmitted: (_) => onSend(),
                ),
              ),
              const SizedBox(width: 10),
              if (isGenerating)
                IconButton.filled(
                  style: IconButton.styleFrom(
                    backgroundColor: const Color(0xFFEF4444),
                    foregroundColor: Colors.white,
                  ),
                  icon: const Icon(Icons.stop_rounded),
                  tooltip: 'Stop generation',
                  onPressed: onStop,
                )
              else
                IconButton.filled(
                  style: IconButton.styleFrom(
                    backgroundColor: const Color(0xFF3B82F6),
                    foregroundColor: Colors.white,
                  ),
                  icon: const Icon(Icons.arrow_upward_rounded),
                  tooltip: 'Send message',
                  onPressed: isBusy ? null : onSend,
                ),
            ],
          ),
        ),
      ],
    );
  }
}
