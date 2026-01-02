import OpenAI from "openai";

const openai =new OpenAI({
    apikey: process.env.Openai_api,
    temperature: 0.7,
    maxTokens: 150
})

console.log(openai.apikey)