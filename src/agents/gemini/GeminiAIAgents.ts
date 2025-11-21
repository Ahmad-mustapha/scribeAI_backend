/// <reference types="node" />
import { GoogleGenAI, Type, Chat, GenerateContentResponse } from "@google/genai";
import type { Channel, StreamChat } from "stream-chat";
import type { AIAgent } from "../types";

/**
 * GeminiAgent
 * ---------------------------------------------------------------
 * This agent listens to Stream Chat `message.new` events
 * and replies using Google's Gemini model.
 * * Features:
 * - Streaming Responses
 * - Web Search (via Tavily API)
 * - UI Theme Control
 * ---------------------------------------------------------------
 */

const DEFAULT_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const GUESTS_MODEL = process.env.GUESTS_MODEL || "gemini-2.5-flash";
const FALLBACK_MODEL = process.env.GEMINI_FALLBACK_MODEL || "gemini-2.5-pro";
const MAX_OUTPUT_TOKENS = Number(process.env.MAX_OUTPUT_TOKENS || 2048);
const MAX_RETRIES = Number(process.env.GEMINI_MAX_RETRIES ?? 2);
const RETRY_DELAY_MS = Number(process.env.GEMINI_RETRY_DELAY_MS ?? 500);

export class GeminiAgent implements AIAgent {
  private genAI!: GoogleGenAI;
  private readonly chatInstances = new Map<string, Chat>();
  // Stores history for cold starts, though Chat instances manage their own history once created
  private readonly conversationHistory = new Map<
    string,
    Array<{ role: "user" | "model"; parts: Array<{ text: string }> }>
  >();
  private lastInteractionTs = Date.now();
  private keepAliveInterval?: NodeJS.Timeout;
  private reconnectAttempts = 0;
  private readonly maxReconnectAttempts = 5;
  private isDisposing = false;
  private lastWatchTime = 0;

  // Helper to delay execution (used in retry/reconnect logic)
  private delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  // Tool Definitions
  private readonly tools = [
    {
      functionDeclarations: [
        {
          name: "web_search",
          description:
            "Search the web for fresh information and return JSON with an answer and sources. Use this when asked about current events, news, or specific facts.",
          parameters: {
            type: Type.OBJECT,
            properties: {
              query: {
                type: Type.STRING,
                description: "Natural-language search query optimized for a search engine",
              },
            },
            required: ["query"],
          },
        },
        {
          name: "change_theme_color",
          description:
            "CRITICAL: This function is REAL and WORKING. You MUST call this function whenever a user requests ANY color change, theme change, or UI appearance modification.",
          parameters: {
            type: Type.OBJECT,
            properties: {
              element: {
                type: Type.STRING,
                description:
                  "The UI element to change. Common values: 'buttons', 'background', 'text', 'borders', 'cards', 'primary', 'secondary'.",
              },
              color: {
                type: Type.STRING,
                description: "The color value (name, hex, rgb, hsl).",
              },
              description: {
                type: Type.STRING,
                description: "A clear description of the change being made.",
              },
            },
            required: ["element", "color", "description"],
          },
        },
      ],
    },
  ] as any;

  constructor(readonly chatClient: StreamChat, readonly channel: Channel) {}

  get user() {
    return this.chatClient.user;
  }

  getLastInteraction = () => this.lastInteractionTs;

  /**
   * Initialize the Gemini client and Stream Chat listeners
   */
  init = async () => {
    try {
      const key = process.env.API_KEY;
    if (!key) {
        throw new Error("API_KEY is required in environment variables.");
      }

      console.log(`[GeminiAgent] Initializing with models: ${DEFAULT_MODEL}`);
      
      // Initialize Google GenAI Client
      this.genAI = new GoogleGenAI({ apiKey: key });

      if (!this.genAI) throw new Error("Failed to create GoogleGenAI client");

      this.setupConnectionHandlers();
      this.lastWatchTime = Date.now();
      this.startKeepAlive();

      // Listen for new messages
      this.chatClient.on("message.new", (event) => {
        this.handleMessage(event);
      });
      
      console.log(`[GeminiAgent] Successfully initialized for channel ${this.channel.id}`);
    } catch (error) {
      console.error(`[GeminiAgent] Failed to initialize:`, error);
      throw error;
    }
  };

  /**
   * EXECUTE WEB SEARCH
   * Uses Tavily API to fetch real-time search results.
   */
  private performWebSearch = async (query: string): Promise<any> => {
    try {
        const apiKey = process.env.TAVILY_API_KEY;
        
        if (!apiKey) {
            console.warn("[GeminiAgent] Web search requested but TAVILY_API_KEY is missing.");
            return { 
                content: "I cannot search the web right now because the search API key is missing in the server configuration." 
            };
        }

        console.log(`[GeminiAgent] Fetching search results for: "${query}"`);

        const response = await fetch("https://api.tavily.com/search", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                api_key: apiKey,
                query: query,
                search_depth: "basic",
                include_answer: true,
                max_results: 3
            })
        });

        if (!response.ok) {
            throw new Error(`Tavily API error: ${response.statusText}`);
        }

        const data = await response.json();
        
        return {
            answer: data.answer, // Direct answer if available
            results: data.results?.map((r: any) => ({
                title: r.title,
                url: r.url,
                content: r.content
            }))
        };
    } catch (e: any) {
        console.error("[GeminiAgent] Search execution failed:", e);
        return { error: `Failed to perform search: ${e.message}` };
    }
  };

  /**
   * Setup connection monitoring
   */
  private setupConnectionHandlers = () => {
    this.chatClient.on("connection.changed", (event: any) => {
      if (event?.online === false && !this.isDisposing) {
            console.warn(`[GeminiAgent] Connection lost, attempting reconnect...`);
        this.handleReconnect();
      } else if (event?.online === true) {
        this.reconnectAttempts = 0;
      }
    });
  };

  /**
   * Handle Reconnection Logic
   */
  private handleReconnect = async () => {
    if (this.isDisposing || this.reconnectAttempts >= this.maxReconnectAttempts) return;
    this.reconnectAttempts++;
    const delayMs = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000);
    await this.delay(delayMs);

    try {
        if (this.chatClient.userID) {
          await this.channel.watch();
          this.reconnectAttempts = 0;
        }
    } catch (error) {
        console.error(`[GeminiAgent] Reconnect failed:`, error);
    }
  };

  /**
   * KeepAlive Heartbeat
   */
  private startKeepAlive = () => {
    if (this.keepAliveInterval) clearInterval(this.keepAliveInterval);
    this.keepAliveInterval = setInterval(async () => {
       if (this.isDisposing) return;
       try {
           if (this.chatClient.userID) {
            const now = Date.now();
               if (now - this.lastWatchTime > 5 * 60 * 1000) {
                await this.channel.watch();
                this.lastWatchTime = now;
               }
           }
       } catch (e) {
           console.error("[GeminiAgent] Keepalive failed", e);
       }
    }, 60000);
  };

  /**
   * Helper to select model based on user type
   */
  private pickModelForUser = (userId?: string) => {
    if (!userId) return DEFAULT_MODEL;
    return userId.startsWith("anon-") ? GUESTS_MODEL : DEFAULT_MODEL;
  };

  /**
   * Retry Wrapper for API Calls
   */
  private performWithRetry = async <T>(
    modelId: string,
    operation: () => Promise<T>
  ): Promise<T> => {
    let lastError: any;
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        return await operation();
      } catch (error: any) {
        lastError = error;
        if (error?.status === 429 || error?.status >= 500) {
           console.warn(`[GeminiAgent] Retry ${attempt}/${MAX_RETRIES} for ${modelId} error: ${error.message}`);
           await this.delay(RETRY_DELAY_MS * attempt);
           continue;
        }
        throw error; 
      }
    }
    throw lastError;
  };

  /**
   * Get or Create Chat Session
   */
  private getOrCreateChatSession(channelId: string, modelId: string): Promise<Chat> {
      let chat = this.chatInstances.get(channelId);
      if (!chat) {
        const systemInstruction = `You are Chromi, an AI writing assistant. 
        Capabilities:
        1. Writing Assistance: Emails, blogs, stories.
        2. Web Search: Use 'web_search' tool for current events/facts.
        3. Theme Customization: Use 'change_theme_color' when users ask to change UI colors.
        Always call the appropriate tool when requested.`;
        
        chat = this.genAI.chats.create({
          model: modelId,
          config: {
                systemInstruction: { parts: [{ text: systemInstruction }] },
            tools: this.tools,
            maxOutputTokens: MAX_OUTPUT_TOKENS,
            temperature: 0.7,
          },
        });
        this.chatInstances.set(channelId, chat);
    }
    return Promise.resolve(chat);
  }
  
  /**
   * Fast Regex Theme Detection (Bypasses LLM for speed/reliability)
   */
  private detectAndExecuteThemeChange = async (userMessage: string) => {
     const lower = userMessage.toLowerCase();
     // Simple heuristic: needs an action word AND a color word/hex
     if (!lower.includes("change") && !lower.includes("set") && !lower.includes("make")) return null;
     
     // This is a placeholder for your regex logic. 
     // The LLM tool is the primary backup if this fails.
     return null; 
  };

  /**
   * MAIN MESSAGE HANDLER
   */
  private handleMessage = async (event: any) => {
      if (!event || event.channel_id !== this.channel.id) return;
      const msg = event.message;
      if (!msg || !msg.text) return;
    if (msg.user?.id === this.user?.id) return; // Ignore own messages

      this.lastInteractionTs = Date.now();

    // 1. Send "Thinking" indicator
      const pendingMessage = await this.channel.sendMessage({
        text: "…",
        custom: { messageType: "ai_response" },
      });

      const userMessageText = String(msg.text);

    // 2. Fast Client-Side Regex Detection (Optional optimization)
    await this.detectAndExecuteThemeChange(userMessageText);

      const desiredModelId = this.pickModelForUser(msg.user?.id);
      const chat = await this.getOrCreateChatSession(event.channel_id, desiredModelId);

      let assistantResponseBuffer = "";
      let lastUpdate = Date.now();

    // Helper to update the Stream Chat UI
      const updateStreamChatMessage = async (final = false) => {
        try {
          await this.chatClient.updateMessage({
            id: pendingMessage.message!.id,
            text: assistantResponseBuffer.trim() || "…",
          } as any);
        } catch (e) {
        console.error("[GeminiAgent] updateMessage failed", e);
        }
      };

      try {
      // 3. Start the conversation stream
      const initialStream = await this.performWithRetry<AsyncGenerator<GenerateContentResponse>>(
        desiredModelId,
        () => chat.sendMessageStream({ message: userMessageText })
      );

      /**
       * RECURSIVE STREAM PROCESSOR
       * Handles the loop: Text -> FunctionCall -> ToolExec -> FunctionResponse -> Text
       */
      const processStream = async (stream: AsyncGenerator<GenerateContentResponse>) => {
        for await (const chunk of stream) {
          // A. Handle Text Content (Streaming to UI)
          if (chunk.text) {
            assistantResponseBuffer += chunk.text;
            // Throttle updates to avoid rate limiting Stream Chat
            if (Date.now() - lastUpdate > 250) {
              await updateStreamChatMessage(false);
              lastUpdate = Date.now();
            }
          }

          // B. Handle Function Calls (Tools)
          if (chunk.functionCalls && chunk.functionCalls.length > 0) {
             for (const fc of chunk.functionCalls) {
                const argObject = fc.args as Record<string, unknown>;
                let toolPayload: any;

                // --- Tool 1: Web Search ---
                if (fc.name === "web_search") {
                    const q = String(argObject?.query ?? "");
                    console.log(`[GeminiAgent] Tool Triggered: web_search '${q}'`);
                    // Execute Search
                    toolPayload = await this.performWebSearch(q);
                }
                
                // --- Tool 2: Change Theme ---
                else if (fc.name === "change_theme_color") {
                    const element = String(argObject?.element ?? "").toLowerCase();
                    const color = String(argObject?.color ?? "");
                    const desc = String(argObject?.description ?? "");
                    console.log(`[GeminiAgent] Tool Triggered: change_theme_color '${element}' -> '${color}'`);

                    // Heuristic to map natural language to CSS vars
                    const elementToVariable: Record<string, string> = {
                        button: "--primary", buttons: "--primary", primary: "--primary",
                        background: "--background", bg: "--background",
                        text: "--foreground", foreground: "--foreground",
                        border: "--border", card: "--card",
                        secondary: "--secondary", destructive: "--destructive"
                    };

                    const variable = Object.keys(elementToVariable).find(k => element.includes(k)) 
                                     ? elementToVariable[Object.keys(elementToVariable).find(k => element.includes(k))!]
                                     : "--primary";

                    // Send specific event to frontend to apply CSS
            await this.channel.sendMessage({
                        text: desc || `Changing ${element} to ${color}`,
                        custom: {
                            messageType: "theme_change",
                            cssVariable: variable,
                            color: color,
                            element: element
                        }
                    });
                    
                    toolPayload = { success: true, message: `Theme changed successfully.` };
                }

                // C. Send Tool Output BACK to Model
                // The model needs the result to generate the final natural language response
                console.log(`[GeminiAgent] Sending tool response back to model...`);
                
                const toolResponseStream = await this.performWithRetry(
                    desiredModelId,
                    () => chat.sendMessageStream({
                        message: [{
                            functionResponse: {
                                name: fc.name,
                                response: toolPayload,
                                id: fc.id // Critical: ID links response to call
                            }
                        }]
                    })
                );

                // Recursively process the NEW stream (the answer derived from the tool)
                await processStream(toolResponseStream);
             }
          }
        }
      };

      // Start processing
      await processStream(initialStream);

      // Final UI Update
      await updateStreamChatMessage(true);

    } catch (error) {
      console.error(`[GeminiAgent] Error processing message:`, error);
      assistantResponseBuffer += "\n\n(I encountered an error processing your request. Please try again.)";
      await updateStreamChatMessage(true);
    }
  };

  dispose = async () => {
    this.isDisposing = true;
    if (this.keepAliveInterval) {
      clearInterval(this.keepAliveInterval);
      this.keepAliveInterval = undefined;
    }
    this.chatClient.off("message.new", this.handleMessage);
    this.chatInstances.clear();
    this.conversationHistory.clear();
    try {
    await this.chatClient.disconnectUser();
    } catch (error) {
      console.error(`[GeminiAgent] Error during disconnect:`, error);
    }
  };
}