// Type definitions for render data
// Python only passes raw data, all processing happens in frontend

export interface Stage {
    name: string
    model: string
    provider: string
    icon_name?: string  // Icon identifier (e.g., "google", "openai")
    icon_config?: string // Config key for icon (e.g. "openai")
    time: number        // Time in seconds (raw number)
    cost: number        // Cost in dollars (raw number)
    usage?: any         // Token usage stats
    references?: Reference[]
    image_references?: ImageReference[]
    crawled_pages?: CrawledPage[]
    description?: string // Brief intro or thought for this stage
    tasks?: string[]
    llm_time?: number   // Time spent in LLM processing
    tool_time?: number  // Time spent in Tool execution
    tool_calls?: number // Number of tool calls

    // Browser JS Driver specific
    url?: string
    script?: string
    output?: string
    js_results?: Array<{
        script: string;
        output: string;
        url?: string;
        success?: boolean;
        error?: string;
    }>
}

export interface Reference {
    title: string
    url: string
    is_fetched?: boolean
    snippet?: string
    type?: string
    images?: string[]  // Extracted images (base64)
    original_idx?: number
    raw_screenshot_b64?: string
    is_thumbnail?: boolean
    screenshot_cache_id?: string
}

export interface ImageReference {
    title: string
    url: string
    thumbnail?: string
}

export interface CrawledPage {
    title: string
    url: string
    description?: string
}

export interface Stats {
    total_time?: number
    vision_duration?: number
    usage?: {
        input_tokens?: number
        cached_input_tokens?: number
        output_tokens?: number
        total_tokens?: number
    }
    operation_rounds?: number
}

export interface Flags {
    has_vision: boolean
    has_search: boolean
}

// Raw data from Python - minimal processing
export interface RenderData {
    markdown: string          // Raw markdown content
    stages: Stage[]
    references: Reference[]   // All references for citation
    page_references: Reference[]
    image_references: ImageReference[]
    stats: Stats
    total_time: number
    theme_color?: string      // Configurable theme color (hex)
}
