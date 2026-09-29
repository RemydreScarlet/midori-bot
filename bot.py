from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import traceback
import uuid
from pathlib import Path

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:8319").rstrip("/")
TTS_MODEL = os.environ.get("TTS_MODEL", "irodori-small")
COMMAND_PREFIX = os.environ.get("TTS_PREFIX", "!")
MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "300"))
MAX_QUEUE = int(os.environ.get("TTS_MAX_QUEUE", "8"))
SYNTH_TIMEOUT = float(os.environ.get("TTS_SYNTH_TIMEOUT", "180"))
TEST_GUILD_ID = int(os.environ.get("DISCORD_GUILD_ID", "0") or 0)

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)

http: aiohttp.ClientSession | None = None
queues: dict[int, asyncio.Queue] = {}
workers: dict[int, asyncio.Task] = {}
guild_options: dict[int, dict] = {}
auto_read: dict[int, int] = {}
_synced = False


async def synth(text: str, options: dict | None = None) -> bytes:
    payload: dict = {"model": TTS_MODEL, "input": text, "response_format": "wav"}
    if options:
        payload["options"] = options
    assert http is not None
    async with http.post(
        f"{TTS_URL}/v1/audio/speech",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=SYNTH_TIMEOUT),
    ) as resp:
        if resp.status != 200:
            body = (await resp.text())[:500]
            raise RuntimeError(f"TTS server {resp.status}: {body}")
        return await resp.read()


async def wait_ready() -> None:
    assert http is not None
    for _ in range(60):
        try:
            async with http.get(f"{TTS_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(1)
    print("[warn] TTS server not reachable yet", flush=True)


async def ensure_connected(guild: discord.Guild, channel: discord.VoiceChannel) -> discord.VoiceClient:
    vc = guild.voice_client
    if isinstance(vc, discord.VoiceClient) and vc.is_connected():
        if vc.channel != channel:
            await vc.move_to(channel)
        return vc
    return await channel.connect(self_deaf=True)


async def connect_voice(ctx: commands.Context) -> discord.VoiceClient | None:
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.reply("先にボイスチャンネルに入ってください。")
        return None
    return await ensure_connected(ctx.guild, ctx.author.voice.channel)


def make_source(path: str) -> discord.AudioSource:
    return discord.FFmpegPCMAudio(path, before_options="-nostdin")


async def play_file(vc: discord.VoiceClient, path: str) -> None:
    loop = asyncio.get_running_loop()
    done = asyncio.Event()

    def after(error: Exception | None) -> None:
        loop.call_soon_threadsafe(done.set)

    if vc.is_playing() or vc.is_paused():
        vc.stop()
    vc.play(make_source(path), after=after)
    await done.wait()


async def worker(guild_id: int) -> None:
    queue = queues[guild_id]
    while True:
        item = await queue.get()
        if item is None:
            break
        text, extra = item
        path = Path(tempfile.gettempdir()) / f"midori_tts_{uuid.uuid4().hex}.wav"
        try:
            data = await synth(text, extra)
            path.write_bytes(data)
            guild = bot.get_guild(guild_id)
            vc = guild.voice_client if guild else None
            if not isinstance(vc, discord.VoiceClient) or not vc.is_connected():
                continue
            await play_file(vc, str(path))
        except Exception as exc:  # noqa: BLE001
            print(f"[tts] {guild_id}: {exc}", flush=True)
            with contextlib.suppress(discord.HTTPException):
                channel = (bot.get_guild(guild_id) and bot.get_guild(guild_id).system_channel)
                if channel:
                    await channel.send(f"TTS失敗: {exc}")
        finally:
            with contextlib.suppress(OSError):
                path.unlink()
            queue.task_done()


def ensure_worker(guild_id: int) -> asyncio.Queue:
    queue = queues.setdefault(guild_id, asyncio.Queue())
    if guild_id not in workers or workers[guild_id].done():
        workers[guild_id] = asyncio.create_task(worker(guild_id))
    return queue


def enqueue(guild_id: int, text: str) -> bool:
    queue = ensure_worker(guild_id)
    if queue.qsize() >= MAX_QUEUE:
        return False
    queue.put_nowait((text, guild_options.get(guild_id, {}).copy()))
    return True


async def clear_queue(guild_id: int) -> None:
    queue = queues.get(guild_id)
    if queue:
        while not queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
                queue.task_done()


@bot.event
async def on_ready() -> None:
    print(f"[bot] logged in as {bot.user} (id={bot.user and bot.user.id})", flush=True)
    await wait_ready()
    global _synced
    if not _synced:
        _synced = True
        if TEST_GUILD_ID:
            guild = discord.Object(id=TEST_GUILD_ID)
            bot.tree.copy_global_to(guild=guild)
            synced = await bot.tree.sync(guild=guild)
        else:
            synced = await bot.tree.sync()
        print(f"[bot] synced {len(synced)} slash commands", flush=True)


@bot.command(name="tts", help="テキストを読み上げます。例: !tts おはようございます")
async def tts(ctx: commands.Context, *, text: str) -> None:
    text = text.strip()
    if not text:
        await ctx.reply("テキストを指定してください。")
        return
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    vc = await connect_voice(ctx)
    if vc is None:
        return

    queue = ensure_worker(ctx.guild.id)
    if not enqueue(ctx.guild.id, text):
        await ctx.reply(f"キューが満です（{MAX_QUEUE}件）。")
        return

    if queue.qsize() == 1:
        await ctx.message.add_reaction("\U0001F50A")
    else:
        await ctx.message.add_reaction("⏳")


@bot.command(name="stop", help="読み上げを止めてキューを空にします")
async def stop(ctx: commands.Context) -> None:
    await clear_queue(ctx.guild.id)
    vc = ctx.voice_client
    if vc and (vc.is_playing() or vc.is_paused()):
        vc.stop()
    await ctx.message.add_reaction("⏹")


@bot.command(name="leave", help="ボイスチャンネルから切断します")
async def leave(ctx: commands.Context) -> None:
    await stop(ctx)
    auto_read.pop(ctx.guild.id, None)
    vc = ctx.voice_client
    if vc:
        await vc.disconnect()
    await ctx.message.add_reaction("👋")


@bot.command(name="voice", help="声の指示を設定します。例: !voice 落ち着いた若い女性。| !voice clear")
async def voice(ctx: commands.Context, *, instruction: str) -> None:
    if instruction.strip().lower() in {"clear", "none", "なし"}:
        guild_options.pop(ctx.guild.id, None)
        await ctx.reply("声の指示を解除しました。")
        return
    guild_options[ctx.guild.id] = {"instruction": instruction.strip()}
    await ctx.reply(f"声の指示を設定: {instruction.strip()}")


@bot.command(name="ping", hidden=True)
async def ping(ctx: commands.Context) -> None:
    await ctx.reply(f"pong (ws: {bot.latency * 1000:.0f}ms)")


@bot.tree.command(name="join", description="ボイスチャンネルに入り、このチャンネルの新着メッセージを読み上げます")
@app_commands.describe(channel="参加するボイスチャンネル（省略時はあなたがいるVC）")
async def join(interaction: discord.Interaction, channel: discord.VoiceChannel | None = None) -> None:
    guild = interaction.guild
    if guild is None:
        await interaction.response.send_message("サーバー内でのみ使えます。", ephemeral=True)
        return

    if channel is None:
        member = interaction.user
        if not isinstance(member, discord.Member) or not member.voice or not member.voice.channel:
            await interaction.response.send_message(
                "先にボイスチャンネルに入ってください。", ephemeral=True
            )
            return
        channel = member.voice.channel

    await interaction.response.defer(ephemeral=True)
    try:
        await ensure_connected(guild, channel)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"[join] failed: {type(exc).__name__}: {exc}", flush=True)
        await interaction.followup.send(
            f"接続失敗: {type(exc).__name__}: {exc}", ephemeral=True
        )
        return

    auto_read[guild.id] = interaction.channel_id
    ensure_worker(guild.id)
    await interaction.followup.send(
        f"{channel.name} に参加しました。このチャンネルのメッセージを読み上げます。"
    )


@bot.tree.command(name="leave", description="ボイスチャンネルから切断し、読み上げを止めます")
async def leave_slash(interaction: discord.Interaction) -> None:
    guild = interaction.guild
    if guild is None:
        await interaction.response.send_message("サーバー内でのみ使えます。", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)
    auto_read.pop(guild.id, None)
    await clear_queue(guild.id)
    vc = guild.voice_client
    if isinstance(vc, discord.VoiceClient):
        if vc.is_playing() or vc.is_paused():
            vc.stop()
        await vc.disconnect()
    await interaction.followup.send("切断しました。")


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.guild is None or message.author.bot:
        await bot.process_commands(message)
        return

    if auto_read.get(message.guild.id) == message.channel.id:
        text = (message.content or "").strip()
        if text and not text.startswith(COMMAND_PREFIX):
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS]
            enqueue(message.guild.id, text)

    await bot.process_commands(message)


@bot.event
async def on_voice_state_update(
    member: discord.Member,
    before: discord.VoiceState,
    after: discord.VoiceState,
) -> None:
    if member.bot or member.guild is None:
        return

    vc = member.guild.voice_client
    if not isinstance(vc, discord.VoiceClient) or not vc.is_connected():
        return

    channel = vc.channel
    if before.channel != channel and after.channel != channel:
        return

    humans = [m for m in channel.members if not m.bot]
    if humans:
        return

    print(f"[voice] alone in {channel.name}, leaving", flush=True)
    auto_read.pop(member.guild.id, None)
    await clear_queue(member.guild.id)
    if vc.is_playing() or vc.is_paused():
        vc.stop()
    with contextlib.suppress(discord.DiscordException):
        await vc.disconnect()


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError) -> None:
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.reply(f"引数不足: {error.param.name}")
        return
    await ctx.reply(f"エラー: {error}")


@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction, error: app_commands.AppCommandError
) -> None:
    traceback.print_exception(type(error), error, error.__traceback__)
    cause = error.original if isinstance(error, app_commands.CommandInvokeError) else error
    msg = f"エラー: {type(cause).__name__}: {cause}"
    with contextlib.suppress(discord.HTTPException):
        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True)


async def main() -> None:
    if not DISCORD_TOKEN:
        raise SystemExit("DISCORD_TOKEN is not set (.env)")
    global http
    http = aiohttp.ClientSession()
    try:
        await bot.start(DISCORD_TOKEN)
    finally:
        await http.close()


if __name__ == "__main__":
    asyncio.run(main())
