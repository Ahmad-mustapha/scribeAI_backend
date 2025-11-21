import { StreamChat } from "stream-chat";
import { apiKey, serverClient } from "../serverClient";
import { AgentPlatform, AIAgent } from "./types";
import { GeminiAgent } from "./gemini/GeminiAIAgents";

export const createAgent = async (
  user_id: string,
  platform: AgentPlatform,
  channel_type: string,
  channel_id: string
): Promise<AIAgent> => {
  console.log(`[createAgent] Starting agent creation for ${user_id}`);
  
  const token = serverClient.createToken(user_id);
  // This is the client for the AI bot user
  const chatClient = new StreamChat(apiKey, undefined, {
    allowServerSideConnect: true,
  });

  console.log(`[createAgent] Connecting user ${user_id}...`);
  await chatClient.connectUser({ id: user_id }, token);
  
  const channel = chatClient.channel(channel_type, channel_id);
  console.log(`[createAgent] Watching channel ${channel_id}...`);
  await channel.watch();
  console.log(`[createAgent] Channel watched successfully`);

  switch (platform) {
    case AgentPlatform.GEMINI:
    case AgentPlatform.WRITING_ASSISTANT:
      console.log(`[createAgent] Creating GeminiAgent instance`);
      return new GeminiAgent(chatClient, channel);
    default:
      throw new Error(`Unsupported agent platform: ${platform}`);
  }
};
