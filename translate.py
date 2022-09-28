import asyncio
import aiohttp
import json

async def translate(text, source, target):
    body= json.dumps({'q':text,'source':source, 'target':target})
    headers= {"Content-Type":"application/json"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(30)) as session:
        async with session.post(url='https://translate.argosopentech.com/translate', data=body, headers=headers) as resp:
            data = await resp.json()
            return data.get('translatedText')

#asyncio.get_event_loop().run_until_complete(translate("で言いたい本当に報告しますはい、ポッ", 'ja', 'en'))