<script setup lang="ts">
import { computed } from 'vue'

import { marked, type Tokens } from 'marked'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import hljs from 'highlight.js/lib/core'
// Import only common languages to reduce bundle size
import python from 'highlight.js/lib/languages/python'
import javascript from 'highlight.js/lib/languages/javascript'
import typescript from 'highlight.js/lib/languages/typescript'
import json from 'highlight.js/lib/languages/json'
import bash from 'highlight.js/lib/languages/bash'
import css from 'highlight.js/lib/languages/css'
import xml from 'highlight.js/lib/languages/xml'
import java from 'highlight.js/lib/languages/java'
import cpp from 'highlight.js/lib/languages/cpp'
import go from 'highlight.js/lib/languages/go'
import rust from 'highlight.js/lib/languages/rust'
import sql from 'highlight.js/lib/languages/sql'
import markdown from 'highlight.js/lib/languages/markdown'
import shell from 'highlight.js/lib/languages/shell'
import yaml from 'highlight.js/lib/languages/yaml'
import properties from 'highlight.js/lib/languages/properties'

hljs.registerLanguage('python', python)
hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('js', javascript)
hljs.registerLanguage('typescript', typescript)
hljs.registerLanguage('ts', typescript)
hljs.registerLanguage('json', json)
hljs.registerLanguage('bash', bash)
hljs.registerLanguage('sh', bash)
hljs.registerLanguage('shell', shell)
hljs.registerLanguage('zsh', bash)
hljs.registerLanguage('css', css)
hljs.registerLanguage('html', xml)
hljs.registerLanguage('xml', xml)
hljs.registerLanguage('java', java)
hljs.registerLanguage('cpp', cpp)
hljs.registerLanguage('c', cpp)
hljs.registerLanguage('go', go)
hljs.registerLanguage('rust', rust)
hljs.registerLanguage('sql', sql)
hljs.registerLanguage('markdown', markdown)
hljs.registerLanguage('md', markdown)
hljs.registerLanguage('yaml', yaml)
hljs.registerLanguage('yml', yaml)
hljs.registerLanguage('properties', properties)
hljs.registerLanguage('ini', properties)
hljs.registerLanguage('conf', properties)

import 'highlight.js/styles/github.css'

const props = defineProps<{
  markdown: string
  numSearchRefs?: number
  numPageRefs?: number
  bare?: boolean
  compact?: boolean
}>()

// Configure marked with syntax highlighting
marked.setOptions({
  breaks: true,
  gfm: true,
})

// Custom renderer for code blocks with technical layout
const renderer = new marked.Renderer()
renderer.code = ({ text, lang }: Tokens.Code): string => {
  const language = lang || 'text'
  let highlighted = ''
  if (lang && hljs.getLanguage(lang)) {
    try {
      highlighted = hljs.highlight(text, { language: lang }).value
    } catch {
      highlighted = hljs.highlightAuto(text).value
    }
  } else {
    highlighted = hljs.highlightAuto(text).value
  }

  // Add line numbers to code
  const addLineNumbers = (code: string): string => {
    const lines = code.split('\n')
    return lines.map((line, i) => 
      `<span class="code-line"><span class="line-number">${i + 1}</span><span class="line-content">${line}</span></span>`
    ).join('')
  }
  
  const highlightedWithLines = addLineNumbers(highlighted)

  // Bare mode: just the code, no window decoration
  if (props.bare) {
    return `<pre class="!mt-0 !mb-0 !rounded-none !bg-gray-50 !p-0 border-b border-gray-100 code-with-lines"><code class="hljs language-${language} text-[17.5px] leading-snug font-mono">${highlightedWithLines}</code></pre>`
  }

  // Dynamic Icon mapping


  return `
    <div class="my-6 space-y-1 group">
      <div class="h-4 w-24 ml-auto" style="background-color: var(--theme-color);"></div>
      <div class="">
        <pre class="!mt-0 !mb-0 !rounded-none !bg-white !p-0 code-with-lines"><code class="hljs language-${language} text-[17.5px] leading-snug font-mono">${highlightedWithLines}</code></pre>
      </div>
    </div>
  `
}

marked.use({ renderer })

// Render LaTeX math with KaTeX
function renderMath(tex: string, displayMode: boolean): string {
  try {
    return katex.renderToString(tex, {
      displayMode,
      throwOnError: false,
      strict: false,
    })
  } catch {
    return `<code>${tex}</code>`
  }
}

// Process markdown and convert citations to badges
const processedHtml = computed(() => {
  let md = props.markdown || ''
  
  // Remove References section at end
  md = md.replace(/(?:^|\n)\s*(?:#{1,3}|\*\*)\s*(?:References|Citations|Sources)[\s\S]*$/i, '')
  
  // Protect math blocks from markdown parsing by replacing with placeholders
  const mathBlocks: { placeholder: string; html: string }[] = []
  let mathIndex = 0
  
  // Block math: $$...$$ or \[...\]
  md = md.replace(/\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]/g, (_, tex1, tex2) => {
    const tex = tex1 || tex2
    const placeholder = `%%MATH_BLOCK_${mathIndex++}%%`
    mathBlocks.push({ placeholder, html: `<div class="my-4 overflow-x-auto">${renderMath(tex.trim(), true)}</div>` })
    return placeholder
  })
  
  // Inline math: $...$ or \(...\) (but not $$)
  md = md.replace(/\$([^\$\n]+?)\$|\\\((.+?)\\\)/g, (_, tex1, tex2) => {
    const tex = tex1 || tex2
    const placeholder = `%%MATH_INLINE_${mathIndex++}%%`
    mathBlocks.push({ placeholder, html: renderMath(tex.trim(), false) })
    return placeholder
  })
  
  // Convert markdown to HTML
  let html = marked.parse(md) as string
  
  // Restore math blocks
  for (const { placeholder, html: mathHtml } of mathBlocks) {
    html = html.replace(placeholder, mathHtml)
  }
  
  // Render <summary> tags as technical highlight blocks
  html = html.replace(/<summary>([\s\S]*?)<\/summary>/g, (_, content) => {
    return `
       <div class="my-8 group shadow-sm shadow-black/10">
        <div class="h-4 w-full" style="background-color: var(--theme-color);"></div>
        <div class="p-6 text-[19px] leading-relaxed font-medium bg-white" style="color: var(--text-body)">
          ${content}
        </div>
      </div>
    `
  })
  
  // Wrap tables in crisp technical borders
  html = html.replace(/<table[^>]*>([\s\S]*?)<\/table>/g, (_, content) => {
    // Parse table content to simple structure
    const rows = content.match(/<tr[^>]*>[\s\S]*?<\/tr>/g) || []
    
    // Extract headers
    const headerRow = rows[0] || ''
    const headers = (headerRow.match(/<th[^>]*>([\s\S]*?)<\/th>/g) || []).map((h: string) => {
      const alignMatch = h.match(/align="([^"]*)"/)
      const align = alignMatch ? alignMatch[1] : 'left'
      const text = h.replace(/<[^>]+>/g, '')
      return { text, align }
    })

    // Extract body rows
    const bodyRows = rows.slice(1).map((row: string) => {
      return (row.match(/<td[^>]*>([\s\S]*?)<\/td>/g) || []).map((c: string, i: number) => {
        const alignMatch = c.match(/align="([^"]*)"/)
        const align = alignMatch ? alignMatch[1] : (headers[i]?.align || 'left') 
        const innerHtml = c.replace(/^<td[^>]*>|<\/td>$/g, '')
        return { html: innerHtml, align }
      })
    })

    const containerClass = "w-full bg-white text-[16.5px] select-text";
      
    let gridHtml = `<div class="${containerClass}">`
    
    type TableCell = { html: string; align: string }
    const allRows: TableCell[][] = [headers.map((h: { text: string; align: string }) => ({ html: h.text, align: h.align })), ...bodyRows]

    allRows.forEach((row, rowIndex) => {
      const isHeader = rowIndex === 0
      const rowBg = isHeader 
        ? 'bg-white text-gray-800 font-black uppercase tracking-tight' 
        : (rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50/30')
      const borderB = rowIndex < allRows.length - 1 ? 'border-b border-gray-200' : ''
        
      gridHtml += `<div class="flex w-full ${rowBg} ${borderB}">`
      
      row.forEach((cell, colIndex) => {
        const justify = cell.align === 'center' ? 'justify-center text-center' : (cell.align === 'right' ? 'justify-end text-right' : 'justify-start');
        const borderClass = colIndex === row.length - 1 ? '' : 'border-r border-gray-100'
        
        gridHtml += `<div class="flex-1 py-3.5 px-4 min-w-0 break-words flex items-center leading-tight ${justify} ${borderClass}">
          <span>${cell.html}</span>
        </div>`
      })
      gridHtml += `</div>`
    })
    gridHtml += `</div>`

    if (props.bare) {
      return `<div class="border-b border-gray-200">${gridHtml}</div>`
    }

    return `
      <div class="my-6 group">
        <div class="bg-white p-0 border-t border-gray-100">
          ${gridHtml}
        </div>
      </div>
    `
  })
  
  // Convert [N] or [N, M] citations to small square badges with shadow
  html = html.replace(/(\s*)\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, _space, content) => {
    return content.split(',').map((n: string) => {
      const num = parseInt(n.trim())
      return `<sup class="inline-flex items-center justify-center w-[15px] h-[15px] text-[10px] font-bold cursor-default select-none ml-0.5 mr-0 align-middle" style="background-color: var(--theme-color); color: var(--header-text-color); box-shadow: 0 1px 2px 0 rgba(0,0,0,0.15)">${num}</sup>`
    }).join('')
  })
  
  // Style <u> underline tags with theme-colored solid underline
  html = html.replace(/<u>([^<]*)<\/u>/g, (_, content) => {
    return `<span class="underline decoration-[3px] underline-offset-[6px]" style="text-decoration-color: var(--theme-color)">${content}</span>`
  })
  
  return html
})
</script>

<template>
  <div ref="contentRef"
       class="prose prose-slate max-w-none"
       :class="props.compact 
         ? 'prose-sm !prose-p:my-1 !prose-p:leading-snug !prose-p:text-[13.5px] !prose-li:text-[13.5px] !prose-headings:text-[14px] !prose-headings:mt-2 !prose-headings:mb-1 prose-a:text-blue-600 prose-a:no-underline hover:prose-a:underline [&>*:first-child]:!mt-0' 
         : 'prose-lg prose-headings:font-bold prose-headings:mb-3 prose-headings:mt-8 prose-headings:tracking-tight prose-p:leading-7 prose-p:my-4 prose-p:text-[20px] prose-li:text-[20px] prose-a:text-blue-600 prose-a:no-underline hover:prose-a:underline prose-code:bg-gray-100 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-none prose-code:text-[0.85em] prose-code:font-mono prose-pre:bg-gray-50 prose-pre:border prose-pre:border-gray-200 prose-pre:rounded-none prose-pre:p-0 prose-img:rounded-none prose-img:my-6 prose-img:max-h-[400px] prose-img:w-auto prose-img:object-contain prose-img:border prose-img:border-gray-200 prose-ol:list-decimal prose-ol:pl-7 prose-ol:list-outside prose-ol:my-5 prose-li:my-2.5 prose-li:leading-7 [&>*:first-child]:!mt-0'"
       style="--prose-headings: var(--text-primary); --prose-body: var(--text-body); --prose-bold: var(--text-primary); --prose-code: var(--text-body)"
       v-html="processedHtml">
  </div>
</template>


<style>
/* Highlight.js theme - minimal */
.hljs {
  background: transparent !important;
  padding: 0 !important;
}

/* Custom List Styling - Premium technical bullet */
.prose ul {
  list-style: none !important;
  padding-left: 0.25rem !important;
  margin-top: 1rem !important;
  margin-bottom: 1rem !important;
}

.prose ul > li {
  position: relative !important;
  padding-left: 1.5rem !important;
  margin-top: 0.75rem !important;
  margin-bottom: 0.75rem !important;
  line-height: 1.6 !important;
}

.prose ul > li::before {
  content: "" !important;
  position: absolute !important;
  left: 0 !important;
  top: 0.6em !important;
  width: 8px !important;
  height: 8px !important;
  background-color: var(--theme-color, #ef4444) !important; /* Theme color */
  border-radius: 0 !important;
}

/* Nested list styling */
.prose ul ul {
  margin-top: 0.25rem !important;
  margin-bottom: 0.25rem !important;
  padding-left: 1rem !important;
}

.prose ul ul > li {
  padding-left: 1.25rem !important;
  margin-top: 0.25rem !important;
  margin-bottom: 0.25rem !important;
}

.prose ul ul > li::before {
  width: 6px !important;
  height: 6px !important;
  background-color: var(--theme-color, #ef4444) !important; /* Theme color - same as parent, slightly smaller */
  top: 0.65em !important;
}

/* Custom Blockquote Styling - Dual Red Lines */
.prose blockquote {
  border-left: none !important;
  padding-left: 1rem !important;
  margin-left: 0 !important;
  position: relative !important;
  font-style: italic !important;
  color: var(--text-body, #3a3a3c) !important; /* Premium reading color */
}

.prose blockquote::before {
  content: "" !important;
  position: absolute !important;
  left: 0 !important;
  top: 0 !important;
  bottom: 0 !important;
  width: 5px !important;
  background-color: var(--theme-color, #ef4444) !important; /* Theme color - thick line */
}



/* Ensure images don't have artifacts */
.prose img {
  display: block;
  margin-left: 0;
  margin-right: auto;
}
.prose pre {
  border: none !important;
}

/* Code line numbers - Modern minimalist style */
.code-with-lines {
  overflow-x: auto;
  overflow-y: hidden;
}
.code-with-lines code {
  display: block;
  width: max-content;
  min-width: 100%;
  padding: 1.25em 0;
  background: white;
}
.code-with-lines .code-line {
  display: flex;
  align-items: stretch;
  min-width: max-content;
}
.code-with-lines .line-number {
  flex-shrink: 0;
  width: 36px;
  padding: 0.1em 8px 0.1em 4px;
  text-align: right;
  color: var(--text-muted, #9ca3af);
  background: white;
  border-right: 1px solid #e5e7eb;
  user-select: none;
  font-size: 11px;
  display: flex;
  align-items: flex-start;
  justify-content: flex-end;
}
.code-with-lines .line-content {
  flex: 0 0 auto;
  min-width: max-content;
  padding: 0.1em 1.25em 0.1em 1em;
  white-space: pre;
  word-break: normal;
  overflow-wrap: normal;
  background: white;
}
</style>
