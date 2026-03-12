<script setup lang="ts">
import { ref, computed, onMounted, nextTick } from 'vue'
import { Icon } from '@iconify/vue'

import type { RenderData, Reference } from './types'
import MarkdownContent from './components/MarkdownContent.vue'

declare global {
  interface Window {
    RENDER_DATA: RenderData
    RENDER_FINISHED: boolean
    updateRenderData: (data: RenderData) => void
  }
}


// Get icon for card type
const getCardIcon = (contentType?: string): string => {
  switch (contentType) {
    case 'summary': return 'mdi:text-box-outline'
    case 'code': return 'mdi:code-braces'
    case 'table': return 'mdi:table'
    default: return 'mdi:card-outline'
  }
}



// Get display label for card
const getCardLabel = (contentType?: string, language?: string): string => {
  switch (contentType) {
    case 'summary': return 'Summary'
    case 'code': return language ? language.charAt(0).toUpperCase() + language.slice(1) : 'Code'
    case 'table': return 'Table'
    default: return ''
  }
}

const data = ref<RenderData | null>(null)

// Expose update method for Python to call
window.updateRenderData = (newData: RenderData) => {
  window.RENDER_FINISHED = false
  data.value = newData
  
  // Wait for rendering to settle
  // Use double nextTick + requestAnimationFrame to ensure Vue has fully rendered
  nextTick(() => {
    nextTick(() => {
      requestAnimationFrame(() => {
        let start = Date.now()
        const check = () => {
          const container = document.getElementById('main-container')
          // If no container yet, keep waiting
          if (!container) {
            if (Date.now() - start > 10000) {
              window.RENDER_FINISHED = true
              return
            }
            setTimeout(check, 100)
            return
          }
          
          const imgs = document.querySelectorAll('img')
          // Check if all images are complete (loaded or errored)
          const allLoaded = imgs.length === 0 || Array.from(imgs).every((img: HTMLImageElement) => img.complete)
          
          // If all images loaded OR timeout (10s)
          if (allLoaded || (Date.now() - start > 10000)) {
            // Extra delay for layout paint (200ms)
            setTimeout(() => { 
              window.RENDER_FINISHED = true 
            }, 200)
          } else {
            setTimeout(check, 100)
          }
        }
        check()
      })
    })
  })
}

const numSearchRefs = computed(() => data.value?.references?.length || 0)
const numPageRefs = computed(() => data.value?.page_references?.length || 0)


// Helper: Strips content before the first H1 heading (e.g., AI "thought" prefixes)
const stripPrefixBeforeH1 = (text: string): string => {
  // Find the first line starting with "# " (H1)
  const h1Match = text.match(/^#\s+/m)
  const summaryMatch = text.match(/<summary>/)
  
  let startIndex = -1
  
  if (h1Match && h1Match.index !== undefined) {
    startIndex = h1Match.index
  }
  
  if (summaryMatch && summaryMatch.index !== undefined) {
    // If summary is found and is BEFORE the H1 (or no H1 found), start from summary
    if (startIndex === -1 || summaryMatch.index < startIndex) {
        startIndex = summaryMatch.index
    }
  }

  if (startIndex !== -1) {
    return text.substring(startIndex)
  }
  
  // If no H1 found, return text as-is (fallback)
  return text
}

// Helper to clean up JS execution context path
// const getJsContextDisplay = (url?: string): string => {
//   if (!url) return 'JavaScript Execution'
//   // Hide local paths
//   if (url.includes('Users') || url.includes('/home/') || url.startsWith('file://')) {
//      return 'VM Context'
//   }
//   return getDomain(url)
// }

// Reorder citations and return cleaned markdown + reordered refs
const reorderedData = computed(() => {
  const originalMd = stripPrefixBeforeH1(data.value?.markdown || '')
  if (!originalMd) return { markdown: '', references: [] }

  const searchRefs = (data.value?.references || []).map((r, i) => ({...r, type: 'search', _orig: i + 1}))
  const pageRefs = (data.value?.page_references || []).map((r, i) => ({...r, type: 'page', _orig: (data.value?.references?.length || 0) + i + 1}))
  const allRefs = [...searchRefs, ...pageRefs]

  const citationRegex = /\[(\d+)\]/g
  const usageOrder: number[] = []
  let match
  // Scan for usage order
  while ((match = citationRegex.exec(originalMd)) !== null) {
    const id = parseInt(match[1]!)
    if (!usageOrder.includes(id)) usageOrder.push(id)
  }
  
  const idMap = new Map<number, number>()
  const newReferences: Reference[] = []
  
  // Only include refs that are actually cited in the markdown
  usageOrder.forEach((oldId, idx) => {
    const newId = idx + 1
    idMap.set(oldId, newId)
    const sourceRef = allRefs[oldId - 1]
    if (sourceRef) newReferences.push({ ...sourceRef, original_idx: newId })
  })
  
  const newMd = originalMd.replace(citationRegex, (m, n) => {
    const newId = idMap.get(parseInt(n))
    return newId ? `[${newId}]` : m
  })
  
  return { markdown: newMd, references: newReferences }
})

const referencesList = computed(() => reorderedData.value.references)

const mainTitle = computed(() => {
  const md = reorderedData.value.markdown || ''
  const match = md.match(/^#\s+(.+)$/m)
  return match && match[1] ? match[1].trim() : ''
})

// Process title to support <u> underline tags
const processedTitle = computed(() => {
  return mainTitle.value.replace(/<u>([^<]*)<\/u>/g, (_, content) => {
    return `<span class="underline decoration-[5px] underline-offset-8" style="text-decoration-color: var(--theme-color)">${content}</span>`
  })
})

function getDomain(url: string): string {
  try {
    const urlObj = new URL(url)
    const hostname = urlObj.hostname.replace('www.', '')
    let pathname = urlObj.pathname === '/' ? '' : decodeURIComponent(urlObj.pathname)
    
    // Truncate if too long
    const maxLen = 40
    let result = hostname + pathname
    if (result.length > maxLen) {
      result = result.slice(0, maxLen - 3) + '...'
    }
    return result
  } catch {
    return url.length > 40 ? url.slice(0, 37) + '...' : url
  }
}

function getFavicon(url: string): string {
  const domain = getDomain(url)
  return `https://www.google.com/s2/favicons?domain=${domain}&sz=32`
}

/**
 * Robustly formats an image source.
 * Handles:
 * 1. Absolute URLs (http, https, //)
 * 2. Data URIs (data:image/...)
 * 3. Raw Base64 strings (fallbacks to data URI)
 */
function getImageUrl(src: string): string {
  if (!src) return ''
  
  // 1. Data URI
  if (src.startsWith('data:')) return src
  
  // 2. Protocol-relative or Absolute URL
  if (src.startsWith('//') || src.startsWith('http:') || src.startsWith('https:')) {
    return src
  }
  
  // 3. Assume raw base64 (remove potential whitespace)
  const cleanBase64 = src.trim()
  if (cleanBase64.length > 0) {
    return `data:image/jpeg;base64,${cleanBase64}`
  }
  
  return src
}

function isValidImage(src: string): boolean {
  if (!src) return false
  if (src.startsWith('http') || src.startsWith('//')) return true
  if (src.length < 20) return false // Too short for meaningful data
  return true
}

// Collect all extracted images from references
const galleryImages = computed(() => {
  const refs = (data.value?.references || []) as Reference[]
  const images: string[] = []
  const seenHashes = new Set<string>()
  
  // Strategy: Balanced picking
  // 1. First Pass: Try to pick 1-2 from each
  for (const ref of refs) {
    if (ref.images && Array.isArray(ref.images)) {
      let count = 0
      for (const b64 of ref.images) {
        if (!isValidImage(b64)) continue
        const hash = `${b64.substring(0, 100)}_${b64.length}`
        if (!seenHashes.has(hash)) {
          seenHashes.add(hash)
          images.push(b64)
          count++
          if (count >= 2) break
        }
      }
    }
  }
  
  // 2. Second Pass: If still too few, pick more from anyone
  if (images.length < 8) {
    for (const ref of refs) {
      if (ref.images && Array.isArray(ref.images)) {
        for (const b64 of ref.images) {
          if (!isValidImage(b64)) continue
          const hash = `${b64.substring(0, 100)}_${b64.length}`
          if (!seenHashes.has(hash)) {
            seenHashes.add(hash)
            images.push(b64)
            if (images.length >= 12) break
          }
        }
      }
      if (images.length >= 12) break
    }
  }

  console.log(`[Gallery] Selected ${images.length} images. First 100 chars of #1:`, images[0]?.substring(0, 100))
  return images.slice(0, 12)
})




const dedent = (text: string) => {
  const lines = text.split('\n')
  // Find minimum indentation of non-empty lines
  let minIndent = Infinity
  for (const line of lines) {
    if (line.trim().length === 0) continue
    const leadingSpace = line.match(/^\s*/)?.[0].length || 0
    if (leadingSpace < minIndent) minIndent = leadingSpace
  }
  
  if (minIndent === Infinity || minIndent === 0) return text
  
  return lines.map(line => {
    if (line.trim().length === 0) return ''
    return line.substring(minIndent)
  }).join('\n')
}


const themeColor = computed(() => data.value?.theme_color || '#ef4444')

// Calculate relative luminance to determine if color is light or dark
const getLuminance = (hex: string): number => {
  const match = hex.replace('#', '').match(/.{2}/g)
  if (!match) return 0
  const [r, g, b] = match.map(x => {
    const c = parseInt(x, 16) / 255
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
  })
  return 0.2126 * (r ?? 0) + 0.7152 * (g ?? 0) + 0.0722 * (b ?? 0)
}

// Auto text color: dark text on light bg, white text on dark bg
const headerTextColor = computed(() => {
  const luminance = getLuminance(themeColor.value)
  return luminance > 0.4 ? '#1f2937' : '#ffffff'  // gray-800 or white
})



const themeStyle = computed(() => ({
  '--theme-color': themeColor.value,
  '--header-text-color': headerTextColor.value,
  '--text-primary': '#2c2c2e',       // Warm dark gray for headings (Apple HIG inspired)
  '--text-body': '#3a3a3c',          // Softer reading color for body text
  '--text-muted': '#86868b',         // Lighter muted secondary text (updated from #636366)
  '--border-color': '#e5e7eb',       // gray-200, for borders
  '--bg-subtle': '#f9fafb'           // gray-50, for subtle backgrounds
}))


const parsedSections = computed(() => {
  const md = reorderedData.value.markdown || ''
  if (!md) return []
  
  let content = md.replace(/^#\s+.+$/m, '')
  content = content.replace(/(?:^|\n)\s*(?:#{1,3}|\*\*)\s*(?:References|Citations|Sources)[\s\S]*$/i, '')
  content = content.trim()
  
  const sections: Array<{ type: 'markdown' | 'card', content: string, title?: string, contentType?: 'table' | 'code' | 'summary', language?: string }> = []
  
  // Combine regex involves complexity, so we'll use a tokenizer approach
  // split tokens by Code Block or Table
  // split tokens by Code Block or Table or Summary
  const combinedRegex = /(```[\s\S]*?```|((?:^|\n)\|[^\n]*\|(?:\n\|[^\n]*\|)*)|<summary>[\s\S]*?<\/summary>)/
  
  let remaining = content
  
  while (remaining) {
    const match = remaining.match(combinedRegex)
    if (!match) {
      if (remaining.trim()) {
        sections.push({ type: 'markdown', content: remaining.trim() })
      }
      break
    }
    
    const index = match.index!
    const matchedStr = match[0]
    const preText = remaining.substring(0, index)
    
    if (preText.trim()) {
      sections.push({ type: 'markdown', content: preText.trim() })
    }
    
    // Determine type
    const isCode = matchedStr.startsWith('```')
    const isSummary = matchedStr.startsWith('<summary>')
    // Tables might match with a leading newline, trim it for checking but render carefully
    const isTable = !isCode && !isSummary && matchedStr.trim().startsWith('|')
    
    if (isCode || isTable || isSummary) {
        let language = ''
        let content = matchedStr.trim()
        
        if (isCode) {
            const match = matchedStr.match(/^```(\w+)/)
            if (match && match[1]) language = match[1]
        } else if (isSummary) {
            // Strip tags
            content = content.replace(/^<summary>/, '').replace(/<\/summary>$/, '')
            content = dedent(content)
        }

        sections.push({
            type: 'card',
            title: isCode ? 'Code' : (isSummary ? 'Summary' : 'Table'),
            content: content,
            contentType: isCode ? 'code' : (isSummary ? 'summary' : 'table'),
            language: language
        })
    } else {
        // Should not happen if regex is correct, but safe fallback
        sections.push({ type: 'markdown', content: matchedStr })
    }
    
    remaining = remaining.substring(index + matchedStr.length)
  }
  
  return sections
})

const formatTokenW = (value?: number): string => {
  const n = Number(value || 0)
  return `${(n / 10000).toFixed(2)}w`
}

const runtimeStatsMarkdown = computed(() => {
  const stats = data.value?.stats || {}
  const usage = stats.usage || {}

  const hasUsage =
    Number(usage.input_tokens || 0) > 0 ||
    Number(usage.cached_input_tokens || 0) > 0 ||
    Number(usage.output_tokens || 0) > 0 ||
    Number(usage.total_tokens || 0) > 0

  if (!hasUsage) {
    return ''
  }

  const rows: string[] = [
    '| Metric | Value |',
    '| --- | --- |',
  ]

  rows.push(
    `| Tokens | ${formatTokenW(usage.total_tokens)} (Input: ${formatTokenW(usage.input_tokens)} / Cached: ${formatTokenW(usage.cached_input_tokens)} / Output: ${formatTokenW(usage.output_tokens)}) |`
  )

  return rows.join('\n')
})

onMounted(() => {
  if (window.RENDER_DATA && Object.keys(window.RENDER_DATA).length > 0) {
    data.value = window.RENDER_DATA
  } else {
    // Demo data for development preview
    data.value = {
      markdown: `# Entari Headless Browser System
This interface demonstrates the capabilities of the Entari Headless Browser system. It generates high-resolution, pixel-perfect captures of AI interactions [1].

<summary>
The system renders Markdown, Code, Tables, and complex UI layouts using a headless browser, optimized for archiving and sharing AI logic flows. This summary block highlights key information [2].
</summary>

## Component Showcase

### Code Highlighting
\`\`\`python
class EntariBrowser:
    def capture(self, url: str) -> bytes:
        """Captures a screenshot of the given URL."""
        return self.driver.get_screenshot_as_png()
\`\`\`

### Data Tables
| Feature | Status | Priority |
| :--- | :--- | :--- |
| Markdown | ✅ Supported | High |
| Syntax Highlight | ✅ Supported | Medium |
| Tables | ✅ Supported | Low |

## Citation Handling
The system automatically handles citations like [1] and [2], reordering them dynamically to match the flow.`,
      total_time: 1.5,
      stages: [
        {
          name: 'instruct',
          model: 'entari/demo-v1',
          provider: 'Entari',
          time: 0.8,
          cost: 0.001,
        },
        {
          name: 'summary',
          model: 'entari/summary-v1',
          provider: 'Entari',
          time: 0.7,
          cost: 0.0005,
        }
      ],
      references: [
        { 
          title: 'Entari Project Documentation', 
          url: 'https://github.com/entari/docs', 
          snippet: 'Official documentation for Entari framework...' 
        },
        { 
          title: 'Headless Browser Concepts', 
          url: 'https://en.wikipedia.org/wiki/Headless_browser', 
          snippet: 'A **headless browser** is a web browser without a graphical user interface.' 
        }
      ],
      page_references: [
        { 
          title: 'Vue.js Framework', 
          url: 'https://vuejs.org/', 
          snippet: 'The Progressive JavaScript Framework. Approachable, Performant, and Versatile.' 
        }
      ],
      image_references: [],
      stats: {
        total_time: 1.5,
        operation_rounds: 2,
        usage: {
          input_tokens: 19893,
          cached_input_tokens: 10368,
          output_tokens: 641,
          total_tokens: 20534,
        },
      },
      theme_color: '#ef4444'
    }
  }
})
</script>

<template>
  <div id="app-wrapper" class="min-h-screen w-full flex justify-center bg-[#f2f2f2]" :style="themeStyle">
    <!-- 
      Container Scaling:
      Width: 560px
      Zoom: 1.5
      Resulting Visual Width: 840px
    -->
    <div class="origin-top my-10" :style="{ zoom: 1.5 }">
      <div id="main-container" class="w-[560px] px-8 py-10 space-y-6 bg-[#f2f2f2]" data-theme="light">
        
        <!-- Title -->
        <header v-if="mainTitle" class="mb-6">
          <h1 class="text-[32px] font-black leading-tight tracking-tighter uppercase tabular-nums" style="color: var(--text-primary)" v-html="processedTitle"></h1>
        </header>

        <!-- Content Sections -->
        <template v-for="(section, idx) in parsedSections" :key="idx">
          
          <!-- Standard Markdown -->
          <div v-if="section.type === 'markdown'">
            <MarkdownContent 
              :markdown="section.content" 
              :num-search-refs="numSearchRefs"
              :num-page-refs="numPageRefs"
              class="prose-h2:text-[22px] prose-h2:font-black prose-h2:uppercase prose-h2:tracking-tight prose-h2:mb-4 prose-h2:text-gray-800"
            />
          </div>

          <!-- Special Card (Table/Code/Summary) -->
          <div v-else-if="section.type === 'card'" class="relative">
            <!-- Corner Rectangle Badge with Icon and Label -->
            <div 
              class="absolute -top-2 -left-2 h-7 px-2.5 z-10 flex items-center justify-center gap-1.5"
              :style="{ backgroundColor: themeColor, color: headerTextColor, boxShadow: '0 2px 4px 0 rgba(0,0,0,0.15)' }"
            >
              <Icon :icon="getCardIcon(section.contentType)" class="text-[14px]" />
              <span class="text-[12px] font-bold uppercase tracking-wide">{{ getCardLabel(section.contentType, section.language) }}</span>
            </div>
            <div 
              class="shadow-sm shadow-black/5 bg-white" 
              :class="[
                section.contentType === 'summary' ? 'pt-8 px-5 pb-4 text-base leading-relaxed break-words' : '',
                section.contentType === 'code' ? 'pt-7 pb-2' : '',
                section.contentType === 'table' ? 'pt-5' : ''
              ]"
            >
              <MarkdownContent 
                :markdown="section.content"
                :bare="true"
                :num-search-refs="numSearchRefs"
                :num-page-refs="numPageRefs"
              />
            </div>
          </div>

        </template>

        <!-- Runtime Stats -->
        <div v-if="runtimeStatsMarkdown" class="relative">
          <div
            class="absolute -top-2 -left-2 h-7 px-2.5 z-10 flex items-center justify-center gap-1.5"
            :style="{ backgroundColor: themeColor, color: headerTextColor, boxShadow: '0 2px 4px 0 rgba(0,0,0,0.15)' }"
          >
            <Icon icon="mdi:counter" class="text-[14px]" />
            <span class="text-[12px] font-bold uppercase tracking-wide">Runtime</span>
          </div>
          <div class="shadow-sm shadow-black/5 bg-white pt-10 px-5 pb-6">
            <MarkdownContent
              :markdown="runtimeStatsMarkdown"
              :bare="true"
              :num-search-refs="0"
              :num-page-refs="0"
            />
          </div>
        </div>
        
        <!-- Sources Section (Bibliography) - Styled as Card -->
        <div v-if="referencesList.length" class="relative group/sources">
          <!-- Corner Rectangle Badge -->
          <div 
            class="absolute -top-2 -left-2 h-7 px-2.5 z-10 flex items-center justify-center gap-1.5"
            :style="{ backgroundColor: themeColor, color: headerTextColor, boxShadow: '0 2px 4px 0 rgba(0,0,0,0.15)' }"
          >
            <Icon icon="mdi:book-open-page-variant-outline" class="text-[14px]" />
            <span class="text-[12px] font-bold uppercase tracking-wide">Sources</span>
          </div>
          
          <div class="shadow-sm shadow-black/5 bg-white pt-10 px-5 pb-6 space-y-6">
             <div v-for="(ref, index) in referencesList" :key="ref.url + '-' + index" class="group/item flex items-start gap-3 pl-0.5">
                <!-- Number -->
                <div class="shrink-0 w-5 h-5 text-[14px] font-bold flex items-center justify-center pt-0.5" 
                     :style="{ color: themeColor }">
                  {{ ref.original_idx }}
                </div>
                
                <!-- Content -->
                <div class="flex-1 min-w-0">
                   <!-- Title -->
                   <a :href="ref.url" target="_blank" class="block mb-0.5">
                     <div class="text-[16px] font-bold leading-tight group-hover/item:text-[var(--theme-color)] transition-colors" style="color: var(--text-primary)">
                       {{ ref.title }}
                     </div>
                   </a>
                   
                   <!-- Domain & Favicon -->
                   <div class="flex items-center gap-2.5 text-[10px] font-mono mb-2" style="color: var(--text-muted)">
                      <img :src="getFavicon(ref.url)" class="w-3 h-3 object-contain rounded-sm">
                      <span>{{ getDomain(ref.url) }}</span>
                   </div>
                   
                   <!-- Snippet / Screenshot (Condition: Must have snippet or raw screenshot) -->
                   <div v-if="ref.raw_screenshot_b64 || ref.snippet"
                        class="mt-1.5 pl-3 py-0.5"
                        :class="[(ref.is_fetched || ref.type === 'page') ? 'border-l-[3px]' : 'border-l-2 border-transparent']"
                        :style="(ref.is_fetched || ref.type === 'page') ? { borderColor: themeColor } : {}"
                   >
                      <!-- Real page screenshot if available -->
                      <div v-if="ref.raw_screenshot_b64" class="relative">
                        <img
                             :src="getImageUrl(ref.raw_screenshot_b64 || '')"
                             class="max-w-full h-auto rounded-sm border border-gray-200 shadow-sm"
                             :class="ref.is_thumbnail ? 'aspect-square object-cover' : ''"
                             alt="Page preview"
                        />
                        <!-- Thumbnail hint -->
                        <div v-if="ref.is_thumbnail && ref.screenshot_cache_id"
                             class="mt-1 text-[10px] font-mono opacity-50"
                             style="color: var(--text-muted)"
                        >
                          /w {{ ref.original_idx }} 查看完整页面
                        </div>
                      </div>
                      <!-- Fallback to markdown snippet -->
                      <MarkdownContent v-else
                        :markdown="ref.snippet || ''"
                        :bare="true"
                        :compact="true"
                      />
                   </div>
                </div>
             </div>
          </div>
        </div>

        <!-- Gallery Section (Extracted Images) - Masonry Layout -->
        <div v-if="galleryImages.length" class="relative group/gallery mb-8">
            <!-- Corner Badge -->
            <div 
              class="absolute -top-2 -left-2 h-7 px-2.5 z-10 flex items-center justify-center gap-1.5"
              :style="{ backgroundColor: themeColor, color: headerTextColor, boxShadow: '0 2px 4px 0 rgba(0,0,0,0.15)' }"
            >
              <Icon icon="mdi:image-multiple-outline" class="text-[14px]" />
              <span class="text-[12px] font-bold uppercase tracking-wide">Gallery</span>
            </div>
            
            <div class="shadow-sm shadow-black/5 bg-white pt-10 px-6 pb-6">
                <!-- Masonry Layout: 2 Columns -->
                <div class="columns-2 gap-4 space-y-4">
                    <div v-for="(img, idx) in galleryImages" :key="idx" class="break-inside-avoid relative rounded-sm overflow-hidden border border-gray-100 bg-gray-50">
                        <img 
                            :src="getImageUrl(img)" 
                            class="w-full h-auto block object-cover transform hover:scale-105 transition-transform duration-500"
                            loading="lazy"
                        />
                    </div>
                </div>
            </div>
        </div>

      </div>
    </div>
  </div>
</template>

<style>
/* Global background fix to prevent white bottom strip */
:root, html, body {
  background-color: #f2f2f2 !important;
  margin: 0;
  padding: 0;
  overflow: hidden !important; /* Force hide scrollbars on root */
  scrollbar-width: none; /* Firefox */
  -ms-overflow-style: none; /* IE and Edge */
}

/* Hide scrollbars for all elements */
*::-webkit-scrollbar {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
}

* {
  scrollbar-width: none !important;
  -ms-overflow-style: none !important;
}
</style>
