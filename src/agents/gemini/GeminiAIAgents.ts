/// <reference types="node" />
import { GoogleGenAI, Type, Chat, GenerateContentResponse } from "@google/genai";
import type { Channel, StreamChat } from "stream-chat";
import type { AIAgent } from "../types";

/**
 * GeminiAgent
 * ---------------------------------------------------------------
 * This agent listens to Stream Chat `message.new` events
 * and replies using Google's Gemini model.
 *
 * - Default model: gemini-2.5-flash  (optimized for streaming: real-time, low latency, cost efficient)
 * - Guest model: gemini-2.5-flash  (for anonymous traffic)
 * - Fallback model: gemini-2.5-pro  (for complex tasks requiring deep analysis)
 * - Supports a single tool: web_search (via Tavily API)
 * - Streams responses live into the chat bubble
 * ---------------------------------------------------------------
 */

// Model selection based on use case:
// - Gemini 2.5 Flash: Best for content streaming (real-time, low latency, cost efficient)
//   Pricing: $0.30/M input tokens, $2.50/M output tokens (paid tier)
// - Gemini 2.5 Flash-Lite: Alternative streaming option (even more cost efficient)
//   Pricing: $0.10/M input tokens, $0.40/M output tokens (paid tier)
// - Gemini 2.5 Pro: Best for complex tasks with deep analysis (higher quality)
// - Gemini 1.5 Flash: Free tier option (if you need free tier)
// 
// Model selection based on use case, updated to recommended models per guidelines:
// - Gemini 2.5 Flash: Best for content streaming (real-time, low latency, cost efficient)
// - Gemini 2.5 Pro: Best for complex tasks with deep analysis (higher quality)
//
// These constants allow overriding via environment variables, falling back to recommended defaults.
const DEFAULT_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const GUESTS_MODEL = process.env.GUESTS_MODEL || "gemini-2.5-flash"; // Using normal flash, not flash-lite
const FALLBACK_MODEL = process.env.GEMINI_FALLBACK_MODEL || "gemini-2.5-pro"; // For complex tasks requiring deep analysis.
const MAX_OUTPUT_TOKENS = Number(process.env.MAX_OUTPUT_TOKENS || 2048);
const MAX_RETRIES = Number(process.env.GEMINI_MAX_RETRIES ?? 2);
const RETRY_DELAY_MS = Number(process.env.GEMINI_RETRY_DELAY_MS ?? 500);
const MAX_HISTORY_LENGTH = Number(process.env.GEMINI_HISTORY_LIMIT ?? 10); // Used for cold-start history

export class GeminiAgent implements AIAgent {
  private genAI!: GoogleGenAI;
  // Use a map to store Chat instances per channel for stateful conversations
  private readonly chatInstances = new Map<string, Chat>();
  // This map stores a basic history to "cold-start" a new Chat instance if one doesn't exist.
  // The Chat object itself manages turns once created.
  private readonly conversationHistory = new Map<
    string,
    Array<{ role: "user" | "model"; parts: Array<{ text: string }> }>
  >();
  private lastInteractionTs = Date.now();
  private keepAliveInterval?: NodeJS.Timeout;
  private reconnectAttempts = 0;
  private readonly maxReconnectAttempts = 5;
  private isDisposing = false;
  private connectionChangedHandler?: (event: any) => void;
  private errorHandler?: (error: any) => void;
  private lastWatchTime = 0;
  private readonly tools = [
    {
      functionDeclarations: [
        {
          name: "web_search",
          description:
            "Search the web for fresh information and return JSON with an answer and sources.",
          parameters: {
            type: Type.OBJECT,
            properties: {
              query: {
                type: Type.STRING,
                description: "Natural-language search query",
              },
            },
            required: ["query"],
          },
        },
        {
          name: "change_theme_color",
          description:
            "CRITICAL: This function is REAL and WORKING. You MUST call this function whenever a user requests ANY color change, theme change, or UI appearance modification. This function directly modifies CSS variables in the running application. Examples that REQUIRE this function: 'change buttons to brown', 'make background blue', 'change text to red', 'set buttons color to #FF5733', 'make borders navy', 'change card color to green'. You have this capability - do NOT refuse or say you cannot change colors. Always call this function for color/theme requests.",
          parameters: {
            type: Type.OBJECT,
            properties: {
              element: {
                type: Type.STRING,
                description:
                  "The UI element to change. Common values: 'buttons' or 'button' (for primary buttons), 'background' or 'bg' (for page background), 'text' or 'foreground' (for text color), 'borders' (for border colors), 'cards' (for card backgrounds), 'primary', 'secondary', 'muted', 'accent', 'input', 'ring', 'destructive', 'popover'. Extract the element name from the user's request.",
              },
              color: {
                type: Type.STRING,
                description:
                  "The color value from the user's request. Accepts: color names (e.g., 'brown', 'navy blue', 'light green', 'dark red'), hex codes (e.g., '#FF5733', '#brown'), RGB (e.g., 'rgb(255,87,51)' or '255, 87, 51'), or HSL (e.g., 'hsl(9,100%,60%)' or '9 100% 60%'). Use exactly what the user specified or a close match.",
              },
              description: {
                type: Type.STRING,
                description:
                  "A clear description of the change being made, based on the user's request (e.g., 'Change all buttons to brown color', 'Set background to navy blue').",
              },
            },
            required: ["element", "color", "description"],
          },
        },
      ] as any,
    },
  ] as any;

  constructor(readonly chatClient: StreamChat, readonly channel: Channel) {}

  get user() {
    return this.chatClient.user;
  }

  getLastInteraction = () => this.lastInteractionTs;

  /**
   * Initialize the Gemini client and subscribe to Stream events
   */
  init = async () => {
    try {
      const key = process.env.API_KEY;
    if (!key) {
        const error = new Error(
          "API_KEY is required. " +
          "Please create a .env file in the backend folder with: API_KEY=your_api_key_here"
        );
        console.error(`[GeminiAgent] Init failed: ${error.message}`);
        console.error(`[GeminiAgent] Env check - API_KEY: ${!!process.env.API_KEY}`);
        throw error;
      }

      console.log(`[GeminiAgent] Initializing with models: ${DEFAULT_MODEL} (default), ${FALLBACK_MODEL} (fallback)`);
      
      this.genAI = new GoogleGenAI({
        apiKey: key,
      });

      // Verify the client was created successfully
      if (!this.genAI) {
        throw new Error("Failed to create GoogleGenAI client");
      }

      // Set up connection monitoring
      this.setupConnectionHandlers();
      this.lastWatchTime = Date.now(); // Initialize watch time
      this.startKeepAlive();

      // Subscribe to message events
      console.log(`[GeminiAgent] Registering message.new event handler for channel ${this.channel.id}`);
      this.chatClient.on("message.new", (event) => {
        console.log(`[GeminiAgent] message.new event received!`);
        this.handleMessage(event);
      });
      
      console.log(`[GeminiAgent] Successfully initialized for channel ${this.channel.id}`);
    } catch (error) {
      console.error(`[GeminiAgent] Failed to initialize:`, error);
      throw error;
    }
  };

  /**
   * Set up connection event handlers for reconnection logic
   */
  private setupConnectionHandlers = () => {
    // Handle disconnection events
    this.connectionChangedHandler = (event: any) => {
      if (event?.online === false && !this.isDisposing) {
        console.warn(
          `[GeminiAgent] Connection lost for ${this.chatClient.user?.id}, attempting reconnect...`
        );
        this.handleReconnect();
      } else if (event?.online === true) {
        console.log(
          `[GeminiAgent] Connection restored for ${this.chatClient.user?.id}`
        );
        this.reconnectAttempts = 0;
      }
    };
    this.chatClient.on("connection.changed", this.connectionChangedHandler);

    // Handle connection errors
    this.errorHandler = (error: any) => {
      console.error(`[GeminiAgent] Stream Chat error:`, error);
      if (!this.isDisposing) {
        this.handleReconnect();
      }
    };
    this.chatClient.on("error", this.errorHandler);
  };

  /**
   * Handle reconnection when connection is lost
   */
  private handleReconnect = async () => {
    if (this.isDisposing || this.reconnectAttempts >= this.maxReconnectAttempts) {
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.error(
          `[GeminiAgent] Max reconnect attempts reached for ${this.chatClient.user?.id}`
        );
      }
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000);

    console.log(
      `[GeminiAgent] Reconnecting (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms...`
    );

    await this.delay(delay);

    try {
      // Check if already connected by checking user ID
      const user = this.chatClient.user;
      if (!user?.id) {
        console.error(`[GeminiAgent] Cannot reconnect: user ID not found`);
        return;
      }

      // Check if client is still connected before using channel
      if (this.chatClient.userID === user.id && this.chatClient.wsConnection) {
        // Try to verify connection is working by re-watching the channel
        try {
          // Try re-watching first (lightweight operation)
          console.log(`[GeminiAgent] Verifying connection by re-watching channel...`);
          await this.channel.watch();
          this.lastWatchTime = Date.now();
          this.reconnectAttempts = 0;
          this.lastInteractionTs = Date.now();
          console.log(`[GeminiAgent] Connection verified, channel re-watched successfully`);
          return;
        } catch (watchError: any) {
          // Re-watch failed, connection might be dead, proceed with full reconnect
          if (watchError?.message?.includes("disconnect")) {
            console.warn(`[GeminiAgent] Client disconnected, proceeding with full reconnect`);
          } else {
            console.warn(`[GeminiAgent] Re-watch failed, proceeding with full reconnect:`, watchError);
          }
        }
      }

      // Import serverClient to create a new token
      const { serverClient } = await import("../../serverClient");
      const token = serverClient.createToken(user.id);

      // Reconnect the client
      await this.chatClient.connectUser({ id: user.id }, token);

      // Wait a bit for connection to stabilize
      await this.delay(100);

      // Re-watch the channel only if client is connected
      if (this.chatClient.userID && this.chatClient.wsConnection) {
        try {
          await this.channel.watch();
          console.log(`[GeminiAgent] Successfully reconnected for ${user.id}`);
          this.reconnectAttempts = 0;
          this.lastInteractionTs = Date.now();
        } catch (channelError: any) {
          console.error(`[GeminiAgent] Failed to re-watch channel after reconnect:`, channelError);
          // Don't throw - connection is restored, channel watch might work on next message
        }
      } else {
        console.warn(`[GeminiAgent] Client not fully connected after reconnect attempt`);
      }
    } catch (error) {
      console.error(`[GeminiAgent] Reconnection failed:`, error);
      // Will retry on next connection.changed event
    }
  };

  /**
   * Start keepalive mechanism to maintain connection
   */
  private startKeepAlive = () => {
    // Clear any existing interval
    if (this.keepAliveInterval) {
      clearInterval(this.keepAliveInterval);
    }

    // Ping every 60 seconds to keep connection alive (less frequent to avoid issues)
    const keepAliveIntervalMs = 60000;

    this.keepAliveInterval = setInterval(async () => {
      if (this.isDisposing) {
        return;
      }

      try {
        // Check if user is connected by verifying user ID exists
        const isConnected = !!this.chatClient.userID;
        const userId = this.chatClient.user?.id;

        if (!isConnected && !this.isDisposing && userId) {
          console.warn(
            `[GeminiAgent] Keepalive detected disconnected state for ${userId}, triggering reconnect...`
          );
          // Reset reconnect attempts for keepalive-triggered reconnects
          this.reconnectAttempts = 0;
          await this.handleReconnect();
          return;
        }

        // Periodic health check - ensure channel is watched and connection is active
        if (isConnected && userId) {
          try {
            // Periodically re-watch the channel to ensure it stays active
            // Re-watch every 5 minutes (300 seconds) to keep connection alive
            const now = Date.now();
            const timeSinceLastWatch = now - this.lastWatchTime;
            const watchInterval = 5 * 60 * 1000; // 5 minutes

            if (timeSinceLastWatch > watchInterval) {
              console.log(`[GeminiAgent] Re-watching channel ${this.channel.id} to maintain connection...`);
              try {
                await this.channel.watch();
                this.lastWatchTime = now;
                this.lastInteractionTs = now;
                console.log(`[GeminiAgent] Successfully re-watched channel ${this.channel.id}`);
              } catch (watchError) {
                console.error(`[GeminiAgent] Failed to re-watch channel:`, watchError);
                // If watch fails, might need to reconnect
                if (!this.isDisposing) {
                  this.reconnectAttempts = 0;
                  await this.handleReconnect();
                }
                return;
              }
            }

            // Lightweight health check: just verify we can access channel state
            // This is much lighter than a full query
            const channelState = this.channel.state;
            if (channelState) {
              this.lastInteractionTs = Date.now();
              // Only log occasionally to avoid spam (every ~10 minutes)
              if (Math.random() < 0.05) {
                console.log(
                  `[GeminiAgent] Keepalive check passed for ${userId} (channel: ${this.channel.id})`
                );
              }
            }
          } catch (error: any) {
            console.error(`[GeminiAgent] Keepalive check failed:`, {
              error: error?.message || String(error),
              userId,
              channelId: this.channel.id,
            });
            // Only reconnect if it's a real connection error, not a transient one
            if (
              error?.message?.includes("connection") ||
              error?.message?.includes("disconnect") ||
              error?.message?.includes("network")
            ) {
              if (!this.isDisposing) {
                // Reset attempts for keepalive-triggered reconnects
                this.reconnectAttempts = 0;
                await this.handleReconnect();
              }
            }
          }
        } else if (!userId) {
          console.warn(`[GeminiAgent] Keepalive: No user ID found, cannot maintain connection`);
        }
      } catch (error) {
        console.error(`[GeminiAgent] Keepalive error:`, error);
      }
    }, keepAliveIntervalMs);
  };

  /**
   * Decide which model to use based on the user ID
   * Uses constants defined at the top.
   */
  private pickModelForUser = (userId?: string) => {
    if (!userId) return DEFAULT_MODEL;
    return userId.startsWith("anon-") ? GUESTS_MODEL : DEFAULT_MODEL;
  };

  /**
   * Gets or creates a new Gemini Chat instance for a given channel.
   * Initializes with stored local history for a "cold start" if available.
   */
  private async getOrCreateChatSession(channelId: string, modelId: string): Promise<Chat> {
    try {
      let chat = this.chatInstances.get(channelId);
      if (!chat) {
        // Use the stored conversation history for initial chat creation.
        // The Chat object then manages its own history for subsequent turns.
        const initialHistory = this.conversationHistory.get(channelId) ?? [];
        console.log(`[GeminiAgent] Creating new Chat session for ${channelId} with model ${modelId}`);
        
        // System instruction that defines the AI's role and capabilities
        const systemInstruction = `You are Chromi, an AI writing assistant that helps users write better content. You are helpful, creative, and professional.

Your capabilities:
1. **Writing Assistance**: Help users write, edit, and improve various types of content including:
   - Business communications (emails, proposals, summaries)
   - Content creation (blog posts, social media, newsletters)
   - Creative writing (stories, characters, brainstorming)
   - Communication improvement (rewriting, tone adjustment)

2. **Theme Customization**: You have REAL access to a change_theme_color function that can modify the UI colors in real-time. When users request ANY color change, theme modification, or UI appearance change, you MUST call this function immediately. Examples:
   - "change buttons to brown" → call change_theme_color
   - "make background blue" → call change_theme_color
   - "set text color to red" → call change_theme_color
   Do NOT refuse or say you cannot change colors - you have this capability.

3. **Web Search**: You can search the web for current information when needed using the web_search function.

Always be helpful, concise, and focus on helping users write better. Use your tools when appropriate.`;
        
        chat = this.genAI.chats.create({
          model: modelId,
          history: initialHistory,
          config: {
            systemInstruction: {
              parts: [{ text: systemInstruction }],
            },
            thinkingConfig: {
              thinkingBudget: -1, // Enable thinking mode like AI Studio
            },
            tools: this.tools,
            maxOutputTokens: MAX_OUTPUT_TOKENS,
            temperature: 0.7,
          },
        });
        
        if (!chat) {
          throw new Error(`Failed to create Chat instance for model ${modelId}`);
        }
        
        this.chatInstances.set(channelId, chat);
        console.log(`[GeminiAgent] Successfully created Chat session for ${channelId}`);
      }
      return chat;
    } catch (error) {
      console.error(`[GeminiAgent] Error creating/getting Chat session for ${channelId}:`, error);
      throw error;
    }
  }

  /**
   * Detect and execute theme changes directly from user messages
   * This bypasses the AI's decision-making for more reliable theme changes
   */
  private detectAndExecuteThemeChange = async (userMessage: string): Promise<{ cssVariable: string; color: string; element: string } | null> => {
    const lowerMessage = userMessage.toLowerCase();
    
    console.log(`[GeminiAgent] Checking message for theme change: "${userMessage}"`);
    
    // More flexible detection - check for action words + UI elements OR color words
    const actionWords = ['change', 'make', 'set', 'update', 'switch', 'turn'];
    const uiElements = ['button', 'background', 'text', 'border', 'card', 'color', 'colour', 'theme'];
    const colorWords = ['brown', 'blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink', 'navy', 'black', 'white', 'gray', 'grey', 'teal', 'cyan', 'magenta', 'lime', 'maroon', 'olive', 'silver', 'gold'];
    
    const hasAction = actionWords.some(word => lowerMessage.includes(word));
    const hasUIElement = uiElements.some(word => lowerMessage.includes(word));
    const hasColor = colorWords.some(word => lowerMessage.includes(word));
    
    // Theme request if: (action + UI element) OR (action + color) OR (UI element + color)
    const hasThemeRequest = (hasAction && hasUIElement) || (hasAction && hasColor) || (hasUIElement && hasColor);
    
    if (!hasThemeRequest) {
      console.log(`[GeminiAgent] No theme request detected`);
      return null;
    }
    
    console.log(`[GeminiAgent] Theme request detected!`);
    
    // Element mapping
    const elementToVariable: Record<string, string> = {
      button: "--primary",
      buttons: "--primary",
      "primary button": "--primary",
      "primary buttons": "--primary",
      primary: "--primary",
      background: "--background",
      bg: "--background",
      "page background": "--background",
      text: "--foreground",
      foreground: "--foreground",
      "font color": "--foreground",
      border: "--border",
      borders: "--border",
      card: "--card",
      cards: "--card",
      secondary: "--secondary",
      "secondary button": "--secondary",
      "secondary buttons": "--secondary",
      muted: "--muted",
      "muted elements": "--muted",
      accent: "--accent",
      "accent color": "--accent",
      input: "--input",
      "input fields": "--input",
      "input field": "--input",
      ring: "--ring",
      "focus ring": "--ring",
      destructive: "--destructive",
      error: "--destructive",
      danger: "--destructive",
      popover: "--popover",
    };
    
    // Extract element name
    let detectedElement = "";
    let detectedColor = "";
    
    // Try to find element in message - prioritize longer matches first
    const sortedKeys = Object.keys(elementToVariable).sort((a, b) => b.length - a.length);
    for (const key of sortedKeys) {
      if (lowerMessage.includes(key)) {
        detectedElement = key;
        console.log(`[GeminiAgent] Found element match: "${key}"`);
        break;
      }
    }
    
    // If no exact match, try partial matches
    if (!detectedElement) {
      const partialMatch = sortedKeys.find((key) =>
        lowerMessage.includes(key.split(' ')[0]) // Match first word
      );
      if (partialMatch) {
        detectedElement = partialMatch;
        console.log(`[GeminiAgent] Found partial element match: "${partialMatch}"`);
      }
    }
    
    // Default to buttons if no element found but has color request
    if (!detectedElement && hasThemeRequest) {
      detectedElement = "buttons";
      console.log(`[GeminiAgent] Defaulting to buttons`);
    }
    
    // Extract color from message - simpler approach
    // First, try to find color words directly in the message
    const allColorWords = ['brown', 'blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink', 'navy', 'black', 'white', 'gray', 'grey', 'teal', 'cyan', 'magenta', 'lime', 'maroon', 'olive', 'silver', 'gold', 'light blue', 'dark blue', 'light green', 'dark green', 'light red', 'dark red'];
    
    // Sort by length (longest first) to match "light blue" before "blue"
    const sortedColorWords = allColorWords.sort((a, b) => b.length - a.length);
    
    for (const colorWord of sortedColorWords) {
      if (lowerMessage.includes(colorWord) && !elementToVariable[colorWord]) {
        detectedColor = colorWord;
        console.log(`[GeminiAgent] Found color word: "${colorWord}"`);
        break;
      }
    }
    
    // If still no color, try regex patterns for hex, RGB, HSL
    if (!detectedColor) {
      const colorPatterns = [
        /#([0-9A-Fa-f]{3,6})\b/i, // Hex codes
        /rgb\(([^)]+)\)/i, // RGB
        /hsl\(([^)]+)\)/i, // HSL
      ];
      
      for (const pattern of colorPatterns) {
        const match = userMessage.match(pattern);
        if (match && match[1]) {
          detectedColor = match[1].trim();
          console.log(`[GeminiAgent] Found color via pattern: "${detectedColor}"`);
          break;
        }
      }
    }
    
    console.log(`[GeminiAgent] Detected element: "${detectedElement}", color: "${detectedColor}"`);
    
    // If we have both element and color, execute the change
    if (detectedElement && detectedColor) {
      const cssVariable = elementToVariable[detectedElement] || "--primary";
      
      console.log(`[GeminiAgent] Executing theme change: ${cssVariable} = ${detectedColor}`);
      
      try {
        // Send theme change message directly
        await this.channel.sendMessage({
          text: `Theme update: Changed ${detectedElement} to ${detectedColor}`,
          custom: {
            messageType: "theme_change",
            cssVariable,
            color: detectedColor,
            element: detectedElement,
            description: `Changed ${detectedElement} to ${detectedColor}`,
          },
        });
        
        console.log(`[GeminiAgent] Successfully sent theme change message`);
        return { cssVariable, color: detectedColor, element: detectedElement };
      } catch (error) {
        console.error(`[GeminiAgent] Failed to send theme change message:`, error);
        return null;
      }
    }
    
    console.log(`[GeminiAgent] Missing element or color - element: ${detectedElement}, color: ${detectedColor}`);
    return null;
  };

  /**
   * Handle an incoming chat message and generate an AI reply
   */
  private handleMessage = async (event: any) => {
    console.log(`[GeminiAgent] handleMessage called with event:`, event?.channel_id, event?.message?.text);
    try {
      if (!event || event.channel_id !== this.channel.id) {
        console.log(`[GeminiAgent] Early return: event check failed`);
        return;
      }

      const msg = event.message;
      if (!msg || !msg.text) {
        console.log(`[GeminiAgent] Early return: no message or text`);
        return;
      }
      if (msg.user?.id === this.user?.id) {
        console.log(`[GeminiAgent] Early return: message from self (${msg.user?.id} === ${this.user?.id})`);
        return; // Don't reply to self or other agent messages
      }

      console.log(`[GeminiAgent] Processing message from user: ${msg.user?.id}, text: "${msg.text}"`);
      this.lastInteractionTs = Date.now();

      // Send a pending message in Stream Chat
      const pendingMessage = await this.channel.sendMessage({
        text: "…",
        custom: { messageType: "ai_response" },
      });

      let userMessageText = String(msg.text);
      
      console.log(`[GeminiAgent] Processing user message: "${userMessageText}"`);
      
      // Direct theme change detection and execution (bypasses AI decision)
      const themeChangeResult = await this.detectAndExecuteThemeChange(userMessageText);
      if (themeChangeResult) {
        // Theme change was applied directly, continue with normal AI response
        console.log(`[GeminiAgent] Applied theme change directly: ${themeChangeResult.cssVariable} = ${themeChangeResult.color}`);
      } else {
        console.log(`[GeminiAgent] No theme change detected or executed`);
      }
      
      const desiredModelId = this.pickModelForUser(msg.user?.id);
      const chat = await this.getOrCreateChatSession(event.channel_id, desiredModelId);

      let assistantResponseBuffer = "";
      let lastUpdate = Date.now();

      // Function to update the Stream Chat message during streaming
      const updateStreamChatMessage = async (final = false) => {
        try {
          await this.chatClient.updateMessage({
            id: pendingMessage.message!.id,
            text: assistantResponseBuffer.trim() || "…",
          } as any);
        } catch (e) {
          console.error("[GeminiAgent] updateMessage failed during streaming", e);
        }
        if (final) {
          lastUpdate = Date.now();
        }
      };

      try {
        // Send the user message to the Gemini Chat and get a streaming response
        // The SDK expects a message parameter
        const streamResult = await this.performWithRetry<AsyncGenerator<GenerateContentResponse>>(
          desiredModelId, // Model ID for logging
          () => chat.sendMessageStream({ message: userMessageText })
        );

        let hasFunctionCall = false;
        
        for await (const chunk of streamResult) {
          // Process function calls
          if (chunk.functionCalls && chunk.functionCalls.length > 0) {
            hasFunctionCall = true;
            
            // Collect all function calls from this chunk
            const functionCalls = chunk.functionCalls;
            
            // Process each function call and send responses
            for (const fc of functionCalls) {
              if (fc.name === "web_search") {
                const argObject = fc.args as Record<string, unknown>;
                const q = String(argObject?.query ?? "");
                console.log(`[GeminiAgent] Performing web_search for query: "${q}"`);
                const toolPayload = await this.performWebSearch(q);

                const toolResponsePart = {
                  functionResponse: {
                    name: "web_search",
                    response: toolPayload,
                    id: fc.id, // Important: include ID for the function response
                  },
                };
                
                // Send the tool response back to the chat model immediately.
                // The SDK expects a message parameter with functionResponse
                // CRITICAL: Function response must come immediately after function call
                console.log(`[GeminiAgent] Sending tool response back to model.`);
                const toolResponseStream = await this.performWithRetry<AsyncGenerator<GenerateContentResponse>>(
                  desiredModelId, // Model ID for logging
                  () => chat.sendMessageStream({
                    message: [{
                      functionResponse: toolResponsePart.functionResponse
                    }]
                  })
                );

                // Process the tool response stream (this is the new conversation turn)
                for await (const toolChunk of toolResponseStream) {
                  // Process text parts (toolChunk.text is a property, not a method)
                  if (toolChunk.text) {
                    assistantResponseBuffer += toolChunk.text;
                    if (Date.now() - lastUpdate > 250) {
                      await updateStreamChatMessage(false);
                      lastUpdate = Date.now();
                    }
                  }
                  
                  // Handle nested function calls (if the model wants to call another function)
                  if (toolChunk.functionCalls && toolChunk.functionCalls.length > 0) {
                    // Recursively handle nested function calls
                    // For now, we'll break and let the outer loop handle it
                    // This is a simplified approach - you may want to handle this recursively
                    console.warn(`[GeminiAgent] Nested function call detected in tool response - this may need special handling`);
                  }
                }
              } else if (fc.name === "change_theme_color") {
                const argObject = fc.args as Record<string, unknown>;
                const element = String(argObject?.element ?? "").toLowerCase();
                const color = String(argObject?.color ?? "");
                const description = String(argObject?.description ?? "");
                
                console.log(`[GeminiAgent] Changing theme color: ${element} to ${color}`);
                
                // Map element names to CSS variables
                const elementToVariable: Record<string, string> = {
                  button: "--primary",
                  buttons: "--primary",
                  "primary button": "--primary",
                  "primary buttons": "--primary",
                  primary: "--primary",
                  background: "--background",
                  bg: "--background",
                  "page background": "--background",
                  text: "--foreground",
                  foreground: "--foreground",
                  "font color": "--foreground",
                  border: "--border",
                  borders: "--border",
                  card: "--card",
                  cards: "--card",
                  secondary: "--secondary",
                  "secondary button": "--secondary",
                  "secondary buttons": "--secondary",
                  muted: "--muted",
                  "muted elements": "--muted",
                  accent: "--accent",
                  "accent color": "--accent",
                  input: "--input",
                  "input fields": "--input",
                  "input field": "--input",
                  ring: "--ring",
                  "focus ring": "--ring",
                  destructive: "--destructive",
                  error: "--destructive",
                  danger: "--destructive",
                  popover: "--popover",
                };
                
                // Find matching CSS variable
                let cssVariable = elementToVariable[element];
                if (!cssVariable) {
                  // Try partial match
                  const partialMatch = Object.keys(elementToVariable).find((key) =>
                    element.includes(key) || key.includes(element)
                  );
                  cssVariable = partialMatch ? elementToVariable[partialMatch] ?? "" : "";
                }
                
                if (!cssVariable) {
                  // Default to primary if no match
                  cssVariable = "--primary";
                  console.warn(`[GeminiAgent] Unknown element "${element}", defaulting to --primary`);
                }
                
                // Send a special message with theme change data
                const themeChangeMessage = await this.channel.sendMessage({
                  text: `Theme update: ${description}`,
                  custom: {
                    messageType: "theme_change",
                    cssVariable,
                    color,
                    element,
                    description,
                  },
                });
                
                console.log(`[GeminiAgent] Sent theme change message: ${cssVariable} = ${color}`);
                
                // Return success response to the model
                const toolResponsePart = {
                  functionResponse: {
                    name: "change_theme_color",
                    response: {
                      success: true,
                      cssVariable,
                      color,
                      element,
                      message: `Successfully changed ${element} color to ${color}`,
                    },
                    id: fc.id,
                  },
                };
                
                const toolResponseStream = await this.performWithRetry<AsyncGenerator<GenerateContentResponse>>(
                  desiredModelId,
                  () => chat.sendMessageStream({
                    message: [{
                      functionResponse: toolResponsePart.functionResponse
                    }]
                  })
                );
                
                // Process the tool response stream
                for await (const toolChunk of toolResponseStream) {
                  if (toolChunk.text) {
                    assistantResponseBuffer += toolChunk.text;
                    if (Date.now() - lastUpdate > 250) {
                      await updateStreamChatMessage(false);
                      lastUpdate = Date.now();
                    }
                  }
                  
                  if (toolChunk.functionCalls && toolChunk.functionCalls.length > 0) {
                    console.warn(`[GeminiAgent] Nested function call detected in theme change response`);
                  }
                }
              }
            }
            
            // CRITICAL: Break out of the original stream loop once we've processed function calls
            // The function response has been sent, and we've processed the tool response stream
            // Continuing to process the original stream would violate the API's requirement
            break;
          }

          // Process text parts (chunk.text is a property, not a method, per guidelines)
          // Only process text if we haven't encountered a function call
          if (!hasFunctionCall && chunk.text) {
            assistantResponseBuffer += chunk.text;
            if (Date.now() - lastUpdate > 250) {
              await updateStreamChatMessage(false);
              lastUpdate = Date.now();
            }
          }
        }

        await updateStreamChatMessage(true); // Final flush

        // Update the local conversation history for potential cold-starts of new Chat sessions.
        // The Chat object itself manages its internal history, this is for agent's state persistence.
        this.updateLocalConversationHistory(event.channel_id, {
            role: "user",
            parts: [{ text: userMessageText }],
        }, assistantResponseBuffer);

        console.log(
          `[GeminiAgent] Final Response for ${event.channel_id} (${desiredModelId}):`,
          assistantResponseBuffer.slice(0, 200)
        );

      } catch (streamError: any) {
        console.error("[GeminiAgent] Error during chat stream processing:", streamError);
        // If we have partial content, try to save it
        if (assistantResponseBuffer.trim().length > 0) {
          await this.chatClient.updateMessage({
            id: pendingMessage.message!.id,
            text: `${assistantResponseBuffer.trim()}\n\n[Note: Response was cut off due to issues]`,
          } as any);
      } else {
            await this.channel.sendMessage({
                text: "Sorry, I ran into an error generating a response. Please try again.",
                custom: { messageType: "system_message" },
            });
        }
      }

    } catch (err) {
      console.error("GeminiAgent.handleMessage overall error", err);
      await this.channel.sendMessage({
        text:
          "Sorry, I ran into an unexpected error handling that message. Please try again in a moment.",
        custom: { messageType: "system_message" },
      });
    }
  };

  /**
   * Refactored performWithRetry to accept a function that directly performs the API call.
   * It only handles retries for the *same* operation on the *same* model/chat instance.
   * Fallback model logic is handled at the `getOrCreateChatSession` level if required.
   */
  private async performWithRetry<T>(
    modelIdForLogging: string, // Used for logging purposes, not for model switching.
    operation: () => Promise<T>
  ): Promise<T> {
    let attempts = 0;
    while (true) {
      try {
        const result = await operation();
        return result;
      } catch (error) {
        if (this.shouldRetry(error) && attempts < MAX_RETRIES) {
          await this.delay(this.retryDelay(attempts));
          attempts += 1;
          console.warn(`[GeminiAgent] Retrying operation for model ${modelIdForLogging} (attempt ${attempts}/${MAX_RETRIES})...`);
          continue;
        }
        console.error(`[GeminiAgent] Operation failed for model ${modelIdForLogging} after ${attempts} retries.`, error);
        throw error;
      }
    }
  }

  private shouldRetry(error: unknown): boolean {
    if (!error || typeof error !== "object") return false;
    const maybe = error as { status?: number; message?: string; name?: string };
    // Retry on 503 Service Unavailable
    if (maybe.status === 503) return true;
    if (typeof maybe.message === "string" && maybe.message.includes("503")) {
      return true;
    }
    // Retry on stream parsing errors (may be transient network issues or incomplete responses)
    if (maybe.name === "GoogleGenAIError") { // Changed from GoogleGenerativeAIError
      const msg = String(maybe.message || "");
      if (msg.includes("Failed to parse stream") || msg.includes("parse stream") || msg.includes("500 Internal Server Error")) {
        return true;
      }
    }
    return false;
  }

  private delay(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private retryDelay(attempt: number) {
    const backoff = RETRY_DELAY_MS * Math.pow(2, attempt);
    return Math.min(backoff, 4000);
  }

  /**
   * Retrieves the current local conversation history for a channel.
   * Used for initializing new Chat instances.
   */
  private getHistory(
    channelId: string
  ): Array<{ role: "user" | "model"; parts: Array<{ text: string }> }> {
    return this.conversationHistory.get(channelId) ?? [];
  }

  /**
   * Updates the local conversation history for cold-starting future chat sessions.
   * This is separate from the `Chat` object's internal history management but ensures
   * the agent remembers past interactions if a chat session is recreated (e.g., due to disconnect).
   */
  private updateLocalConversationHistory(
    channelId: string,
    userEntry: { role: "user"; parts: Array<{ text: string }> },
    assistantText: string
  ) {
    if (!assistantText) return;
    let history = this.conversationHistory.get(channelId) ?? [];
    history.push(userEntry);
    history.push({
      role: "model",
      parts: [{ text: assistantText }],
    });
    const trimmed =
      history.length > MAX_HISTORY_LENGTH
        ? history.slice(history.length - MAX_HISTORY_LENGTH)
        : history;
    this.conversationHistory.set(channelId, trimmed);
  }

  /**
   * Perform a web search via Tavily API
   */
  private performWebSearch = async (query: string) => {
    const TAVILY_API_KEY = process.env.TAVILY_API_KEY;
    if (!TAVILY_API_KEY) {
      return { error: "Web search unavailable: TAVILY_API_KEY not set." };
    }

    try {
      const res = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${TAVILY_API_KEY}`,
        },
        body: JSON.stringify({
          query,
          search_depth: "advanced",
          max_results: 5,
          include_answer: true,
          include_raw_content: false,
        }),
      });

      if (!res.ok) {
        return { error: `Tavily error: ${res.status} ${res.statusText}` };
      }
      return await res.json();
    } catch (e: any) {
      return { error: "Search exception", message: e?.message ?? String(e) };
    }
  };

  dispose = async () => {
    this.isDisposing = true;

    // Clear keepalive interval
    if (this.keepAliveInterval) {
      clearInterval(this.keepAliveInterval);
      this.keepAliveInterval = undefined;
    }

    // Remove event listeners
    this.chatClient.off("message.new", this.handleMessage);
    if (this.connectionChangedHandler) {
      this.chatClient.off("connection.changed", this.connectionChangedHandler);
    }
    if (this.errorHandler) {
      this.chatClient.off("error", this.errorHandler);
    }

    // Clear all active chat sessions (instances will be garbage collected)
    this.chatInstances.clear();

    // Disconnect user
    try {
    await this.chatClient.disconnectUser();
    } catch (error) {
      console.error(`[GeminiAgent] Error during disconnect:`, error);
    }
  };
}
// agents/gemini/GeminiAgent.ts


