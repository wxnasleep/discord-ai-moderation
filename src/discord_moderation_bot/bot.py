"""Discord message listener and moderation actions."""

import logging
from datetime import timedelta

import discord
from discord.ext import commands

from .config import Settings
from .moderation import ToxicityModerator
from .storage import WarningStore


logger = logging.getLogger(__name__)


class ModerationBot(commands.Bot):
    def __init__(self, settings: Settings):
        intents = discord.Intents.default()
        intents.guilds = True
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

        self.settings = settings
        self.moderator: ToxicityModerator | None = None
        self.warning_store = WarningStore(settings.data_path)

    async def setup_hook(self) -> None:
        self.warning_store.initialize()
        self.moderator = ToxicityModerator.load(self.settings.model_path)
        if self.settings.guild_id:
            guild = discord.Object(id=self.settings.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()

    async def on_ready(self) -> None:
        logger.info("Signed in as %s; moderation listener is ready", self.user)

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.guild is None:
            return

        await self.process_commands(message)
        if self.moderator is None or not message.content.strip():
            return
        if self.settings.guild_id and message.guild.id != self.settings.guild_id:
            return

        score = await self.moderator.check(message.content)
        if score < self.settings.warning_threshold:
            return

        if score >= self.settings.severe_threshold:
            await self._handle_severe(message, score)
            return

        warning_number = self.warning_store.add_warning(message.guild.id, message.author.id)
        if warning_number >= self.settings.warnings_before_timeout:
            timeout_applied = await self._timeout_member(
                message, score, self.settings.warning_timeout_minutes
            )
            self.warning_store.clear_warnings(message.guild.id, message.author.id)
            await self._send_warning(message, score, warning_number, timed_out=timeout_applied)
            await self._report(
                message,
                score,
                f"{warning_number} warnings; timeout {'applied' if timeout_applied else 'failed'}",
            )
            return
        await self._send_warning(message, score, warning_number)

    async def _handle_severe(self, message: discord.Message, score: int) -> None:
        member = message.author
        assert isinstance(member, discord.Member)

        try:
            await message.delete()
        except discord.HTTPException:
            logger.info("Could not delete a severe message in guild %s", message.guild.id)

        timeout_applied = await self._timeout_member(
            message, score, self.settings.severe_timeout_minutes
        )
        await self._report(
            message,
            score,
            f"Severe message deleted; timeout {'applied' if timeout_applied else 'failed'}",
        )

    async def _timeout_member(self, message: discord.Message, score: int, minutes: int) -> bool:
        member = message.author
        assert isinstance(member, discord.Member)
        try:
            await member.timeout(
                timedelta(minutes=minutes),
                reason=f"Automatic moderation: model score {score}/10",
            )
            return True
        except (discord.Forbidden, discord.HTTPException):
            logger.warning("Could not timeout member %s", member.id)
            return False

    async def _send_warning(
        self, message: discord.Message, score: int, count: int, timed_out: bool = False
    ) -> None:
        if timed_out:
            text = (
                f"{message.author.display_name}: предупреждение {count}/"
                f"{self.settings.warnings_before_timeout}. Выдан timeout на "
                f"{self.settings.warning_timeout_minutes} мин."
            )
        else:
            text = (
                f"{message.author.display_name}: предупреждение {count}/"
                f"{self.settings.warnings_before_timeout} "
                f"(оценка модерации: {score}/10)."
            )
        try:
            await message.channel.send(text, allowed_mentions=discord.AllowedMentions.none())
        except discord.HTTPException:
            logger.warning("Could not send warning in guild %s", message.guild.id)

    async def _report(self, message: discord.Message, score: int, action: str) -> None:
        if self.settings.moderation_channel_id:
            destination = self.get_channel(self.settings.moderation_channel_id)
            if destination is None:
                try:
                    destination = await self.fetch_channel(self.settings.moderation_channel_id)
                except discord.HTTPException:
                    logger.warning("Configured moderation channel is unavailable")
                    return
        else:
            destination = message.channel
        if destination is None or not hasattr(destination, "send"):
            logger.warning("Moderation channel is unavailable")
            return

        embed = discord.Embed(
            title="AI moderation",
            description=f"Action: **{action}**\nScore: **{score}/10**",
            color=discord.Color.orange() if score < self.settings.severe_threshold else discord.Color.red(),
        )
        embed.add_field(name="User", value=f"{message.author.mention} (`{message.author.id}`)", inline=False)
        embed.add_field(name="Channel", value=message.channel.mention, inline=True)
        embed.add_field(name="Message", value=discord.utils.escape_markdown(message.content[:900]) or "(empty)", inline=False)
        try:
            await destination.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())
        except discord.HTTPException:
            logger.exception("Failed to send moderation report")


def create_bot(settings: Settings) -> ModerationBot:
    return ModerationBot(settings)
