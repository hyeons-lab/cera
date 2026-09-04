export type PlatformId = 'rust' | 'web' | 'flutter' | 'android' | 'apple' | 'python' | 'cli';

export interface CodeSnippet {
  language: string;
  code: string;
  filename?: string;
  description?: string;
}

export interface DocSection {
  id: string;
  title: string;
  content: string;
  codeSnippets?: Partial<Record<PlatformId, CodeSnippet>>;
  subsections?: Array<{
    id: string;
    title: string;
    content: string;
    code?: CodeSnippet;
  }>;
}

export interface DocCategory {
  id: string;
  title: string;
  description: string;
  iconName: string;
  articles: DocArticle[];
}

export interface DocArticle {
  id: string;
  title: string;
  description: string;
  badge?: string;
  sections: DocSection[];
}

export interface LeapBundleModel {
  id: string;
  name: string;
  architecture: string;
  parameters: string;
  modalities: string[];
  quants: string[];
  contextLength: string;
  description: string;
  huggingFaceUrl: string;
  featured?: boolean;
}

export interface BenchmarkEntry {
  device: string;
  chip: string;
  backend: string;
  model: string;
  quant: string;
  prefillTokPerSec: number;
  decodeTokPerSec: number;
  memoryMb: number;
  ttftMs: number;
}
