import 'dotenv/config';
import OpenAI from "openai";

const openai =new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
})

// console.log(openai.apiKey);
// console.log(process.env.OPENAI_API_KEY);


const messages = [
        {
            role: "user",
            content: "Hello, how are you?"
        }
    ]

const response = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: messages
})


console.log(response)