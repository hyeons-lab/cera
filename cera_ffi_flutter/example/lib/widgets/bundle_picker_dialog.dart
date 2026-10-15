import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../app_constants.dart';
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
  static const _kDialogTitle = 'Select Model';
  static const _kTabDownloaded = 'Downloaded';
  static const _kTabCatalog = 'Catalog & Download';
  static const _kActionCancel = 'Cancel';
  static const _kActionLoad = 'Load';
  static const _kActionRemoveTooltip = 'Remove from list';
  static const _kEmptyDownloadedTitle = 'No models downloaded yet';
  static const _kEmptyDownloadedSubtitle =
      'Download models from the catalog to run fast, offline on-device inference.';
  static const _kBrowseCatalogButton = 'Browse Catalog to Download';
  static const _kSearchHint =
      'Filter models (e.g. VL, Audio, 1.2B, Thinking)...';
  static const _kBadgeActive = 'Active';
  static const _kBadgeDownloaded = 'Downloaded';
  static const _kBadgeDSpark = 'DSpark';
  static const _kBadgeDSparkSidecar = 'DSpark Sidecar';
  static const _kBadgeVision = 'Vision';
  static const _kBadgeAudio = 'Audio';
  static const _kBadgeMoe = 'MoE';
  static const _kBadgeThinking = 'Thinking';
  static const _kMonospaceFont = 'monospace';
  static const _kAmberAccent = Color(0xFFF59E0B);

  static const _kStandardQuants = ['Q4_K_M', 'Q4_0', 'Q8_0', 'F16'];
  static const _kCompactQuants = ['Q4_K_M', 'Q8_0', 'F16'];
  static const _kAudioQuants = ['Q4_0', 'Q8_0', 'F16'];
  static const _kDSparkSupportedQuants = {'Q4_K_M', 'Q8_0', 'F16'};

  static const _kDefaultBundles = [
    CeraBundle(name: 'LFM2.5-2.6B-GGUF', quants: _kStandardQuants),
    CeraBundle(name: 'LFM2.5-2.6B-Agent-GGUF', quants: _kStandardQuants),
    CeraBundle(name: 'LFM2.5-2.6B-Thinking-GGUF', quants: _kStandardQuants),
    CeraBundle(name: 'LFM2.5-1.2B-Instruct-GGUF', quants: _kStandardQuants),
    CeraBundle(name: 'LFM2.5-1.2B-Thinking-GGUF', quants: _kStandardQuants),
    CeraBundle(name: 'LFM2.5-8B-A1B-GGUF', quants: _kStandardQuants),
    CeraBundle(name: 'LFM2.5-8B-A1B-Instruct-GGUF', quants: _kStandardQuants),
    CeraBundle(name: 'LFM2.5-700M-GGUF', quants: _kStandardQuants),
    CeraBundle(name: 'LFM2.5-350M-GGUF', quants: _kStandardQuants),
    CeraBundle(name: 'LFM2.5-230M-GGUF', quants: _kStandardQuants),
    CeraBundle(name: 'LFM2.5-VL-3B-GGUF', quants: _kCompactQuants),
    CeraBundle(name: 'LFM2.5-Audio-1.5B-GGUF', quants: _kAudioQuants),
    CeraBundle(name: 'LFM2-1.2B-GGUF', quants: _kStandardQuants),
    CeraBundle(name: 'LFM2-700M-GGUF', quants: _kStandardQuants),
    CeraBundle(name: 'LFM2-350M-GGUF', quants: _kStandardQuants),
    CeraBundle(name: 'LFM2-2.6B-Exp-GGUF', quants: _kCompactQuants),
  ];

  late final Future<List<CeraBundle>> _bundles = _fetchBundles();
  final TextEditingController _searchController = TextEditingController();
  String _searchQuery = '';
  List<DownloadedModelRecord> _downloaded = [];
  bool _loadingDownloaded = true;
  _PickerTab _tab = _PickerTab.downloaded;

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<List<CeraBundle>> _fetchBundles() async {
    List<CeraBundle> live = [];
    try {
      live = await Cera.listBundles();
    } catch (_) {}

    final map = {for (final b in live) b.name: b};

    if (map.isEmpty) {
      for (final b in _kDefaultBundles) {
        map[b.name] = b;
      }
    }

    final list = map.values.toList();
    final current = widget.currentBundleName;
    list.sort((a, b) {
      if (current != null) {
        final aIsCurrent = a.name == current;
        final bIsCurrent = b.name == current;
        if (aIsCurrent != bIsCurrent) return aIsCurrent ? -1 : 1;
      }
      return a.displayName.toLowerCase().compareTo(b.displayName.toLowerCase());
    });
    return list;
  }

  bool _hasHfDSparkDraft(String bundleName, String quant) {
    final clean = quant.split(RegExp(r'[+ ]')).first.trim().toUpperCase();
    final name = bundleName.toLowerCase();
    if (name.contains('lfm2.5-2.6b') ||
        name.contains('lfm2.5-1.2b') ||
        name.contains('lfm2.5-8b')) {
      return _kDSparkSupportedQuants.contains(clean);
    }
    return false;
  }

  @override
  void initState() {
    super.initState();
    _loadDownloadedModels();
  }

  Future<void> _loadDownloadedModels() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final list =
          prefs.getStringList(PrefKeys.downloadedModels) ?? const <String>[];
      final records = <DownloadedModelRecord>[];
      for (final s in list) {
        try {
          final m = jsonDecode(s) as Map<String, dynamic>;
          records.add(DownloadedModelRecord.fromJson(m));
        } catch (_) {}
      }

      // If active model is loaded but not yet registered in downloaded records, include it
      final currentBundle = widget.currentBundleName;
      final currentQuant = widget.currentQuant;
      if (currentBundle != null && currentQuant != null) {
        final activeId = '$currentBundle:$currentQuant';
        if (!records.any((r) => r.id == activeId)) {
          records.insert(
            0,
            DownloadedModelRecord(
              bundleName: currentBundle,
              quant: currentQuant,
              displayName: ModelFormat.stripGgufSuffix(currentBundle),
            ),
          );
        }
      }

      if (mounted) {
        setState(() {
          _downloaded = records;
          _loadingDownloaded = false;
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
      await prefs.setStringList(PrefKeys.downloadedModels, list);
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return AlertDialog(
      backgroundColor: theme.colorScheme.surface,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: theme.colorScheme.outline),
      ),
      title: const Text(
        _kDialogTitle,
        style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
      ),
      content: SizedBox(
        width: 460,
        height: 480,
        child: Column(
          children: [
            Container(
              decoration: BoxDecoration(
                color: theme.scaffoldBackgroundColor,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: theme.colorScheme.outline),
              ),
              padding: const EdgeInsets.all(3),
              child: Row(
                children: [
                  Expanded(
                    child: _TabButton(
                      label: _kTabDownloaded,
                      count: _downloaded.length,
                      isSelected: _tab == _PickerTab.downloaded,
                      onTap: () => setState(() => _tab = _PickerTab.downloaded),
                    ),
                  ),
                  Expanded(
                    child: _TabButton(
                      label: _kTabCatalog,
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
          child: const Text(_kActionCancel),
        ),
      ],
    );
  }

  Widget _buildDownloadedView(ThemeData theme) {
    if (_loadingDownloaded) {
      return Center(
        child: CircularProgressIndicator(
          valueColor: AlwaysStoppedAnimation<Color>(theme.colorScheme.primary),
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
              Icon(
                Icons.folder_open_rounded,
                size: 48,
                color: theme.colorScheme.onSurfaceVariant,
              ),
              const SizedBox(height: 12),
              Text(
                _kEmptyDownloadedTitle,
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  fontSize: 15,
                  color: theme.colorScheme.onSurface,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                _kEmptyDownloadedSubtitle,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: theme.colorScheme.onSurfaceVariant,
                  fontSize: 12,
                  height: 1.4,
                ),
              ),
              const SizedBox(height: 18),
              FilledButton.icon(
                style: FilledButton.styleFrom(
                  backgroundColor: theme.colorScheme.primary,
                  foregroundColor: theme.colorScheme.onPrimary,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 10,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                icon: const Icon(Icons.cloud_download_outlined, size: 16),
                label: const Text(_kBrowseCatalogButton),
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
          Divider(color: theme.colorScheme.outlineVariant, height: 1),
      itemBuilder: (context, i) {
        final model = _downloaded[i];
        final isActive =
            model.bundleName == widget.currentBundleName &&
            model.quant == widget.currentQuant;

        return Material(
          color: isActive
              ? theme.colorScheme.primary.withValues(alpha: 0.16)
              : Colors.transparent,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
            side: isActive
                ? BorderSide(
                    color: theme.colorScheme.primary.withValues(alpha: 0.4),
                  )
                : BorderSide.none,
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
                      color: isActive ? theme.colorScheme.primary : null,
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
                      color: theme.colorScheme.primaryContainer.withValues(
                        alpha: 0.5,
                      ),
                      borderRadius: BorderRadius.circular(6),
                      border: Border.all(
                        color: theme.colorScheme.primary.withValues(alpha: 0.4),
                      ),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.check_circle_rounded,
                          size: 11,
                          color: theme.colorScheme.primary,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          _kBadgeActive,
                          style: TextStyle(
                            color: theme.colorScheme.primary,
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
            subtitle: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  model.quant,
                  style: TextStyle(
                    fontFamily: _kMonospaceFont,
                    fontSize: 12,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
                if (model.quant.toLowerCase().contains('dspark')) ...[
                  const SizedBox(width: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 5,
                      vertical: 1,
                    ),
                    decoration: BoxDecoration(
                      color: _kAmberAccent.withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(4),
                      border: Border.all(
                        color: _kAmberAccent.withValues(alpha: 0.4),
                      ),
                    ),
                    child: const Text(
                      _kBadgeDSpark,
                      style: TextStyle(
                        color: _kAmberAccent,
                        fontSize: 9,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ],
              ],
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
                    child: const Text(
                      _kActionLoad,
                      style: TextStyle(fontSize: 12),
                    ),
                    onPressed: () => Navigator.of(context).pop(
                      BundleChoice(
                        bundleName: model.bundleName,
                        quant: model.quant,
                        displayName: model.displayName,
                      ),
                    ),
                  )
                else
                  Icon(
                    Icons.check_rounded,
                    size: 18,
                    color: theme.colorScheme.primary,
                  ),
                const SizedBox(width: 4),
                IconButton(
                  icon: Icon(
                    Icons.delete_outline_rounded,
                    size: 18,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                  tooltip: _kActionRemoveTooltip,
                  onPressed: () => _removeDownloadedModel(model),
                ),
              ],
            ),
            onTap: () {
              if (isActive) {
                Navigator.of(context).pop();
                return;
              }
              Navigator.of(context).pop(
                BundleChoice(
                  bundleName: model.bundleName,
                  quant: model.quant,
                  displayName: model.displayName,
                ),
              );
            },
          ),
        );
      },
    );
  }

  static String? _modalityBadgeFor(String bundleName) {
    final lower = bundleName.toLowerCase();
    if (lower.contains('-vl-')) return _kBadgeVision;
    if (lower.contains('-audio-')) return _kBadgeAudio;
    if (lower.contains('-8b-a1b')) return _kBadgeMoe;
    if (lower.contains('-thinking')) return _kBadgeThinking;
    return null;
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
        final allBundles = snapshot.data;
        if (allBundles == null) {
          return Center(
            child: CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation<Color>(
                theme.colorScheme.primary,
              ),
            ),
          );
        }
        final q = _searchQuery.trim().toLowerCase();
        final bundles = q.isEmpty
            ? allBundles
            : allBundles.where((b) {
                final mod = _modalityBadgeFor(b.name)?.toLowerCase() ?? '';
                return b.displayName.toLowerCase().contains(q) ||
                    b.name.toLowerCase().contains(q) ||
                    mod.contains(q);
              }).toList();

        return Column(
          children: [
            TextField(
              controller: _searchController,
              onChanged: (val) => setState(() => _searchQuery = val),
              style: TextStyle(
                fontSize: 13,
                color: theme.colorScheme.onSurface,
              ),
              decoration: InputDecoration(
                hintText: _kSearchHint,
                hintStyle: TextStyle(
                  fontSize: 12.5,
                  color: theme.colorScheme.onSurfaceVariant,
                ),
                prefixIcon: Icon(
                  Icons.search_rounded,
                  size: 18,
                  color: theme.colorScheme.onSurfaceVariant,
                ),
                suffixIcon: _searchQuery.isNotEmpty
                    ? IconButton(
                        icon: Icon(
                          Icons.close_rounded,
                          size: 16,
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                        onPressed: () {
                          _searchController.clear();
                          setState(() => _searchQuery = '');
                        },
                      )
                    : null,
                isDense: true,
                filled: true,
                fillColor: theme.scaffoldBackgroundColor,
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 10,
                ),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: BorderSide(color: theme.colorScheme.outline),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: BorderSide(color: theme.colorScheme.outline),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: BorderSide(color: theme.colorScheme.primary),
                ),
              ),
            ),
            const SizedBox(height: 10),
            Expanded(
              child: bundles.isEmpty
                  ? Center(
                      child: Text(
                        'No models match "$_searchQuery"',
                        style: TextStyle(
                          fontSize: 13,
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                    )
                  : ListView.separated(
                      itemCount: bundles.length,
                      separatorBuilder: (_, _) => Divider(
                        color: theme.colorScheme.outlineVariant,
                        height: 1,
                      ),
                      itemBuilder: (context, i) {
                        final bundle = bundles[i];
                        final isCurrentBundle =
                            bundle.name == widget.currentBundleName;
                        final modalityBadge = _modalityBadgeFor(bundle.name);

                        final quantOptions = <String>[];
                        int dsparkCount = 0;
                        for (final q in bundle.quants) {
                          quantOptions.add(q);
                          if (_hasHfDSparkDraft(bundle.name, q)) {
                            quantOptions.add('$q + $_kBadgeDSpark');
                            dsparkCount++;
                          }
                        }
                        final n = bundle.quants.length;

                        return ExpansionTile(
                          key: PageStorageKey(bundle.name),
                          initiallyExpanded: isCurrentBundle,
                          title: Row(
                            children: [
                              Expanded(
                                child: Text(
                                  bundle.displayName,
                                  style: const TextStyle(
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ),
                              if (modalityBadge != null) ...[
                                const SizedBox(width: 6),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 6,
                                    vertical: 1.5,
                                  ),
                                  decoration: BoxDecoration(
                                    color: theme.colorScheme.primary.withValues(
                                      alpha: 0.14,
                                    ),
                                    borderRadius: BorderRadius.circular(4),
                                    border: Border.all(
                                      color: theme.colorScheme.primary
                                          .withValues(alpha: 0.35),
                                    ),
                                  ),
                                  child: Text(
                                    modalityBadge,
                                    style: TextStyle(
                                      color: theme.colorScheme.primary,
                                      fontSize: 10,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                                ),
                              ],
                              if (bundle.name.toLowerCase().contains(
                                    'dspark',
                                  ) ||
                                  dsparkCount > 0) ...[
                                const SizedBox(width: 6),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 6,
                                    vertical: 1.5,
                                  ),
                                  decoration: BoxDecoration(
                                    color: _kAmberAccent.withValues(alpha: 0.2),
                                    borderRadius: BorderRadius.circular(4),
                                    border: Border.all(
                                      color: _kAmberAccent.withValues(
                                        alpha: 0.4,
                                      ),
                                    ),
                                  ),
                                  child: const Text(
                                    _kBadgeDSpark,
                                    style: TextStyle(
                                      color: _kAmberAccent,
                                      fontSize: 10,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                                ),
                              ],
                              if (isCurrentBundle)
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 7,
                                    vertical: 2,
                                  ),
                                  decoration: BoxDecoration(
                                    color: theme.colorScheme.primaryContainer
                                        .withValues(alpha: 0.5),
                                    borderRadius: BorderRadius.circular(6),
                                    border: Border.all(
                                      color: theme.colorScheme.primary
                                          .withValues(alpha: 0.4),
                                    ),
                                  ),
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Icon(
                                        Icons.check_circle_rounded,
                                        size: 12,
                                        color: theme.colorScheme.primary,
                                      ),
                                      const SizedBox(width: 4),
                                      Text(
                                        _kBadgeActive,
                                        style: TextStyle(
                                          color: theme.colorScheme.primary,
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
                            dsparkCount > 0
                                ? '$n base quants${AppStrings.metaSeparator}$dsparkCount with $_kBadgeDSpark sidecar'
                                : '$n quantization${n == 1 ? '' : 's'}',
                            style: TextStyle(
                              color: theme.colorScheme.onSurfaceVariant,
                              fontSize: 12,
                            ),
                          ),
                          children: [
                            for (final quant in quantOptions)
                              Builder(
                                builder: (context) {
                                  final isLoadedQuant =
                                      isCurrentBundle &&
                                      quant == widget.currentQuant;
                                  final isDownloaded = _downloaded.any(
                                    (r) =>
                                        r.bundleName == bundle.name &&
                                        r.quant == quant,
                                  );
                                  final isDSpark = quant.contains(
                                    _kBadgeDSpark,
                                  );
                                  final baseQuant = isDSpark
                                      ? quant.split(' ').first
                                      : quant;

                                  return Material(
                                    color: isLoadedQuant
                                        ? theme.colorScheme.primary.withValues(
                                            alpha: 0.16,
                                          )
                                        : Colors.transparent,
                                    child: ListTile(
                                      dense: true,
                                      title: Row(
                                        children: [
                                          Text(
                                            baseQuant,
                                            style: TextStyle(
                                              fontFamily: _kMonospaceFont,
                                              fontWeight: isLoadedQuant
                                                  ? FontWeight.w700
                                                  : FontWeight.normal,
                                              color: isLoadedQuant
                                                  ? theme.colorScheme.primary
                                                  : null,
                                            ),
                                          ),
                                          if (isDSpark) ...[
                                            const SizedBox(width: 8),
                                            Container(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                    horizontal: 6,
                                                    vertical: 1.5,
                                                  ),
                                              decoration: BoxDecoration(
                                                color: _kAmberAccent.withValues(
                                                  alpha: 0.2,
                                                ),
                                                borderRadius:
                                                    BorderRadius.circular(4),
                                                border: Border.all(
                                                  color: _kAmberAccent
                                                      .withValues(alpha: 0.4),
                                                ),
                                              ),
                                              child: const Text(
                                                _kBadgeDSparkSidecar,
                                                style: TextStyle(
                                                  color: _kAmberAccent,
                                                  fontSize: 10,
                                                  fontWeight: FontWeight.w700,
                                                ),
                                              ),
                                            ),
                                          ],
                                          if (isLoadedQuant) ...[
                                            const SizedBox(width: 8),
                                            Container(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                    horizontal: 6,
                                                    vertical: 1,
                                                  ),
                                              decoration: BoxDecoration(
                                                color: theme.colorScheme.primary
                                                    .withValues(alpha: 0.2),
                                                borderRadius:
                                                    BorderRadius.circular(4),
                                              ),
                                              child: Text(
                                                _kBadgeActive,
                                                style: TextStyle(
                                                  color:
                                                      theme.colorScheme.primary,
                                                  fontSize: 10,
                                                  fontWeight: FontWeight.w700,
                                                ),
                                              ),
                                            ),
                                          ] else if (isDownloaded) ...[
                                            const SizedBox(width: 8),
                                            Container(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                    horizontal: 6,
                                                    vertical: 1,
                                                  ),
                                              decoration: BoxDecoration(
                                                color: theme
                                                    .colorScheme
                                                    .outlineVariant,
                                                borderRadius:
                                                    BorderRadius.circular(4),
                                              ),
                                              child: Text(
                                                _kBadgeDownloaded,
                                                style: TextStyle(
                                                  color: theme
                                                      .colorScheme
                                                      .onSurfaceVariant,
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
                                            ? theme.colorScheme.primary
                                            : theme
                                                  .colorScheme
                                                  .onSurfaceVariant,
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
                    ),
            ),
          ],
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
    final theme = Theme.of(context);
    final itemCount = count ?? 0;
    return Material(
      color: isSelected ? theme.colorScheme.outlineVariant : Colors.transparent,
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
                      ? theme.colorScheme.onSurface
                      : theme.colorScheme.onSurfaceVariant,
                ),
                const SizedBox(width: 6),
              ],
              Flexible(
                child: Text(
                  label,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: isSelected ? FontWeight.w600 : FontWeight.w500,
                    color: isSelected
                        ? theme.colorScheme.onSurface
                        : theme.colorScheme.onSurfaceVariant,
                  ),
                ),
              ),
              if (itemCount > 0) ...[
                const SizedBox(width: 6),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 6,
                    vertical: 1,
                  ),
                  decoration: BoxDecoration(
                    color: isSelected
                        ? theme.colorScheme.primary
                        : theme.colorScheme.outlineVariant,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    '$itemCount',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: isSelected
                          ? theme.colorScheme.onPrimary
                          : theme.colorScheme.onSurfaceVariant,
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
