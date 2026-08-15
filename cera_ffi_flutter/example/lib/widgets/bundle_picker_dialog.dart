import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../chat_state.dart';

/// A bundle choice carrying identification and display properties.
class BundleChoice {
  const BundleChoice({
    required this.bundleName,
    required this.quant,
    required this.displayName,
  });

  final String bundleName;
  final String quant;
  final String displayName;
}

enum _PickerTab { downloaded, catalog }

/// Two-state dialog for choosing between downloaded models and downloading new ones.
class BundlePickerDialog extends StatefulWidget {
  const BundlePickerDialog({
    super.key,
    this.currentBundleName,
    this.currentQuant,
  });

  final String? currentBundleName;
  final String? currentQuant;

  @override
  State<BundlePickerDialog> createState() => _BundlePickerDialogState();
}

class _BundlePickerDialogState extends State<BundlePickerDialog> {
  late final Future<List<CeraBundle>> _bundles = Cera.listBundles();
  List<DownloadedModelRecord> _downloaded = [];
  bool _loadingDownloaded = true;
  _PickerTab _tab = _PickerTab.downloaded;

  @override
  void initState() {
    super.initState();
    _loadDownloadedModels();
  }

  Future<void> _loadDownloadedModels() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final list = prefs.getStringList('cera_downloaded_models') ?? [];
      final records = <DownloadedModelRecord>[];
      for (final s in list) {
        try {
          final m = jsonDecode(s) as Map<String, dynamic>;
          records.add(DownloadedModelRecord.fromJson(m));
        } catch (_) {}
      }

      // If active model is loaded but not yet registered in downloaded records, include it
      if (widget.currentBundleName != null && widget.currentQuant != null) {
        final activeId = '${widget.currentBundleName}:${widget.currentQuant}';
        if (!records.any((r) => r.id == activeId)) {
          final name = widget.currentBundleName!;
          final display = name.endsWith('-GGUF')
              ? name.substring(0, name.length - '-GGUF'.length)
              : name;
          records.insert(
            0,
            DownloadedModelRecord(
              bundleName: name,
              quant: widget.currentQuant!,
              displayName: display,
            ),
          );
        }
      }

      if (mounted) {
        setState(() {
          _downloaded = records;
          _loadingDownloaded = false;
          // If no models are downloaded yet, open catalog by default
          if (records.isEmpty) {
            _tab = _PickerTab.catalog;
          }
        });
      }
    } catch (_) {
      if (mounted) setState(() => _loadingDownloaded = false);
    }
  }

  Future<void> _removeDownloadedModel(DownloadedModelRecord record) async {
    setState(() {
      _downloaded.removeWhere((r) => r.id == record.id);
    });
    try {
      final prefs = await SharedPreferences.getInstance();
      final list = _downloaded.map((r) => jsonEncode(r.toJson())).toList();
      await prefs.setStringList('cera_downloaded_models', list);
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return AlertDialog(
      backgroundColor: const Color(0xFF14161B),
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: const BorderSide(color: Color(0xFF232732)),
      ),
      title: const Text(
        'Select Model',
        style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
      ),
      content: SizedBox(
        width: 460,
        height: 480,
        child: Column(
          children: [
            // Segmented Tab Switcher
            Container(
              decoration: BoxDecoration(
                color: const Color(0xFF0B0C0E),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: const Color(0xFF232732)),
              ),
              padding: const EdgeInsets.all(3),
              child: Row(
                children: [
                  Expanded(
                    child: _TabButton(
                      label: 'Downloaded',
                      count: _downloaded.length,
                      isSelected: _tab == _PickerTab.downloaded,
                      onTap: () => setState(() => _tab = _PickerTab.downloaded),
                    ),
                  ),
                  Expanded(
                    child: _TabButton(
                      label: 'Catalog & Download',
                      icon: Icons.cloud_download_outlined,
                      isSelected: _tab == _PickerTab.catalog,
                      onTap: () => setState(() => _tab = _PickerTab.catalog),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 14),
            Expanded(
              child: _tab == _PickerTab.downloaded
                  ? _buildDownloadedView(theme)
                  : _buildCatalogView(theme),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
      ],
    );
  }

  Widget _buildDownloadedView(ThemeData theme) {
    if (_loadingDownloaded) {
      return const Center(
        child: CircularProgressIndicator(
          valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF3B82F6)),
        ),
      );
    }

    if (_downloaded.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(
                Icons.folder_open_rounded,
                size: 48,
                color: Color(0xFF64748B),
              ),
              const SizedBox(height: 12),
              const Text(
                'No models downloaded yet',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  fontSize: 15,
                  color: Color(0xFFF1F5F9),
                ),
              ),
              const SizedBox(height: 6),
              const Text(
                'Download models from the catalog to run fast, offline on-device inference.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Color(0xFF8E95A5),
                  fontSize: 12,
                  height: 1.4,
                ),
              ),
              const SizedBox(height: 18),
              FilledButton.icon(
                style: FilledButton.styleFrom(
                  backgroundColor: const Color(0xFF3B82F6),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 10,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                icon: const Icon(Icons.cloud_download_outlined, size: 16),
                label: const Text('Browse Catalog to Download'),
                onPressed: () => setState(() => _tab = _PickerTab.catalog),
              ),
            ],
          ),
        ),
      );
    }

    return ListView.separated(
      itemCount: _downloaded.length,
      separatorBuilder: (_, _) =>
          const Divider(color: Color(0xFF1E222D), height: 1),
      itemBuilder: (context, i) {
        final model = _downloaded[i];
        final isActive =
            model.bundleName == widget.currentBundleName &&
            model.quant == widget.currentQuant;

        return Container(
          decoration: BoxDecoration(
            color: isActive ? const Color(0xFF162338) : null,
            borderRadius: BorderRadius.circular(8),
            border: isActive
                ? Border.all(color: const Color(0x663B82F6), width: 1)
                : null,
          ),
          child: ListTile(
            dense: true,
            title: Row(
              children: [
                Expanded(
                  child: Text(
                    model.displayName,
                    style: TextStyle(
                      fontWeight: isActive ? FontWeight.w700 : FontWeight.w600,
                      color: isActive ? const Color(0xFF93C5FD) : null,
                    ),
                  ),
                ),
                if (isActive)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 7,
                      vertical: 2,
                    ),
                    decoration: BoxDecoration(
                      color: const Color(0xFF0E3E2F),
                      borderRadius: BorderRadius.circular(6),
                      border: Border.all(color: const Color(0x6610B981)),
                    ),
                    child: const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.check_circle_rounded,
                          size: 11,
                          color: Color(0xFF34D399),
                        ),
                        SizedBox(width: 4),
                        Text(
                          'Active',
                          style: TextStyle(
                            color: Color(0xFF34D399),
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
            subtitle: Text(
              model.quant,
              style: const TextStyle(
                fontFamily: 'monospace',
                fontSize: 12,
                color: Color(0xFF8E95A5),
              ),
            ),
            trailing: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (!isActive)
                  FilledButton.tonal(
                    style: FilledButton.styleFrom(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      minimumSize: Size.zero,
                      tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    ),
                    child: const Text('Load', style: TextStyle(fontSize: 12)),
                    onPressed: () => Navigator.of(context).pop(
                      BundleChoice(
                        bundleName: model.bundleName,
                        quant: model.quant,
                        displayName: model.displayName,
                      ),
                    ),
                  )
                else
                  const Icon(
                    Icons.check_rounded,
                    size: 18,
                    color: Color(0xFF60A5FA),
                  ),
                const SizedBox(width: 4),
                IconButton(
                  icon: const Icon(
                    Icons.delete_outline_rounded,
                    size: 18,
                    color: Color(0xFF64748B),
                  ),
                  tooltip: 'Remove from list',
                  onPressed: () => _removeDownloadedModel(model),
                ),
              ],
            ),
            onTap: () => Navigator.of(context).pop(
              BundleChoice(
                bundleName: model.bundleName,
                quant: model.quant,
                displayName: model.displayName,
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildCatalogView(ThemeData theme) {
    return FutureBuilder<List<CeraBundle>>(
      future: _bundles,
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return Center(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Text(
                'Could not reach the catalog:\n${snapshot.error}',
                style: TextStyle(color: theme.colorScheme.error),
              ),
            ),
          );
        }
        final rawBundles = snapshot.data;
        if (rawBundles == null) {
          return const Center(
            child: CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF3B82F6)),
            ),
          );
        }
        final bundles = rawBundles.toList()
          ..sort(
            (a, b) => a.displayName.toLowerCase().compareTo(
              b.displayName.toLowerCase(),
            ),
          );
        return ListView.separated(
          itemCount: bundles.length,
          separatorBuilder: (_, _) =>
              const Divider(color: Color(0xFF1E222D), height: 1),
          itemBuilder: (context, i) {
            final bundle = bundles[i];
            final isCurrentBundle = bundle.name == widget.currentBundleName;
            final quants = bundle.quants;
            final n = quants.length;
            return ExpansionTile(
              key: PageStorageKey(bundle.name),
              initiallyExpanded: isCurrentBundle,
              title: Row(
                children: [
                  Expanded(
                    child: Text(
                      bundle.displayName,
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                  ),
                  if (isCurrentBundle)
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 7,
                        vertical: 2,
                      ),
                      decoration: BoxDecoration(
                        color: const Color(0xFF0E3E2F),
                        borderRadius: BorderRadius.circular(6),
                        border: Border.all(color: const Color(0x6610B981)),
                      ),
                      child: const Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.check_circle_rounded,
                            size: 12,
                            color: Color(0xFF34D399),
                          ),
                          SizedBox(width: 4),
                          Text(
                            'Active',
                            style: TextStyle(
                              color: Color(0xFF34D399),
                              fontSize: 11,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                ],
              ),
              subtitle: Text(
                '$n quantization${n == 1 ? "" : "s"}',
                style: const TextStyle(color: Color(0xFF8E95A5), fontSize: 12),
              ),
              children: [
                for (final quant in quants)
                  Builder(
                    builder: (context) {
                      final isLoadedQuant =
                          isCurrentBundle && quant == widget.currentQuant;
                      final isDownloaded = _downloaded.any(
                        (r) => r.bundleName == bundle.name && r.quant == quant,
                      );
                      return Container(
                        color: isLoadedQuant ? const Color(0xFF162338) : null,
                        child: ListTile(
                          dense: true,
                          title: Row(
                            children: [
                              Text(
                                quant,
                                style: TextStyle(
                                  fontFamily: 'monospace',
                                  fontWeight: isLoadedQuant
                                      ? FontWeight.w700
                                      : FontWeight.normal,
                                  color: isLoadedQuant
                                      ? const Color(0xFF93C5FD)
                                      : null,
                                ),
                              ),
                              if (isLoadedQuant) ...[
                                const SizedBox(width: 8),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 6,
                                    vertical: 1,
                                  ),
                                  decoration: BoxDecoration(
                                    color: const Color(0xFF1E3A5F),
                                    borderRadius: BorderRadius.circular(4),
                                  ),
                                  child: const Text(
                                    'Active',
                                    style: TextStyle(
                                      color: Color(0xFF60A5FA),
                                      fontSize: 10,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                                ),
                              ] else if (isDownloaded) ...[
                                const SizedBox(width: 8),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 6,
                                    vertical: 1,
                                  ),
                                  decoration: BoxDecoration(
                                    color: const Color(0xFF1F2937),
                                    borderRadius: BorderRadius.circular(4),
                                  ),
                                  child: const Text(
                                    'Downloaded',
                                    style: TextStyle(
                                      color: Color(0xFF9CA3AF),
                                      fontSize: 10,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ),
                              ],
                            ],
                          ),
                          trailing: Icon(
                            isLoadedQuant
                                ? Icons.check_rounded
                                : (isDownloaded
                                      ? Icons.play_arrow_rounded
                                      : Icons.download_rounded),
                            size: 18,
                            color: isLoadedQuant
                                ? const Color(0xFF60A5FA)
                                : const Color(0xFF94A3B8),
                          ),
                          onTap: () => Navigator.of(context).pop(
                            BundleChoice(
                              bundleName: bundle.name,
                              quant: quant,
                              displayName: bundle.displayName,
                            ),
                          ),
                        ),
                      );
                    },
                  ),
              ],
            );
          },
        );
      },
    );
  }
}

class _TabButton extends StatelessWidget {
  const _TabButton({
    required this.label,
    required this.isSelected,
    required this.onTap,
    this.count,
    this.icon,
  });

  final String label;
  final bool isSelected;
  final VoidCallback onTap;
  final int? count;
  final IconData? icon;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: isSelected ? const Color(0xFF1E222D) : Colors.transparent,
      borderRadius: BorderRadius.circular(7),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(7),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 8),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (icon != null) ...[
                Icon(
                  icon,
                  size: 15,
                  color: isSelected
                      ? const Color(0xFFF1F5F9)
                      : const Color(0xFF8E95A5),
                ),
                const SizedBox(width: 6),
              ],
              Text(
                label,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: isSelected ? FontWeight.w600 : FontWeight.w500,
                  color: isSelected
                      ? const Color(0xFFF1F5F9)
                      : const Color(0xFF8E95A5),
                ),
              ),
              if (count != null && count! > 0) ...[
                const SizedBox(width: 6),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 6,
                    vertical: 1,
                  ),
                  decoration: BoxDecoration(
                    color: isSelected
                        ? const Color(0xFF3B82F6)
                        : const Color(0xFF262938),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    '$count',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: isSelected
                          ? Colors.white
                          : const Color(0xFF94A3B8),
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
