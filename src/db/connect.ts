// db/connect.ts
import mongoose from "mongoose";

export async function connectDB() {
    try {
        if (mongoose.connection.readyState >= 1) return;
        const uri = process.env.MONGO_URI;
        if (!uri) {
            throw new Error("MONGO_URI is not set");
        }
        await mongoose.connect(uri);
        console.log("✅ MongoDB connected");
    } catch (error) {
        console.error("❌ MongoDB connection error", error);
        process.exit(1);
    }
}
