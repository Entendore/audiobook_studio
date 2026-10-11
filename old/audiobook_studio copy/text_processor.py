"""
Streaming text extraction, cleaning, chapter detection, and chunking.

Includes the professional 10-pass SpeakerDetectionEngine for
multi-voice audiobooks, SSML generation support, pronunciation
dictionary integration, reading-level analysis, pronunciation
auto-suggest, and improved scene-break chapter detection.

No Qt dependency — pure Python 3.10+.

Optional dependencies: spaCy (en_core_web_sm), pypdf, python-docx,
beautifulsoup4, ebooklib.
"""

from __future__ import annotations

import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Dict,
    FrozenSet,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
)

# ---------------------------------------------------------------------------
# Optional spaCy dependency
# ---------------------------------------------------------------------------
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    _SPACY_OK = True
except Exception:
    _NLP = None
    _SPACY_OK = False

# ---------------------------------------------------------------------------
# Optional dependencies (Document Parsing)
# ---------------------------------------------------------------------------
try:
    from pypdf import PdfReader
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from docx import Document
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

try:
    from ebooklib import epub
    EPUB_OK = True
except ImportError:
    EPUB_OK = False

logger = logging.getLogger(__name__)


# ===================================================================
# 1. SPEAKER ENGINE CONSTANTS
# ===================================================================

SPEECH_VERBS: FrozenSet[str] = frozenset({
    "said", "says", "say", "saying",
    "asked", "asks", "ask", "asking",
    "replied", "replies", "reply", "replying",
    "answered", "answers", "answer", "answering",
    "whispered", "whispers", "whisper", "whispering",
    "shouted", "shouts", "shout", "shouting",
    "cried", "cries", "cry", "crying",
    "exclaimed", "exclaims", "exclaim", "exclaiming",
    "murmured", "murmurs", "murmur", "murmuring",
    "muttered", "mutters", "mutter", "muttering",
    "gasped", "gasps", "gasp", "gasping",
    "sighed", "sighs", "sigh", "sighing",
    "laughed", "laughs", "laugh", "laughing",
    "chuckled", "chuckles", "chuckle", "chuckling",
    "snarled", "snarls", "snarl", "snarling",
    "growled", "growls", "growl", "growling",
    "hissed", "hisses", "hiss", "hissing",
    "barked", "barks", "bark", "barking",
    "yelled", "yells", "yell", "yelling",
    "screamed", "screams", "scream", "screaming",
    "called", "calls", "call", "calling",
    "declared", "declares", "declare", "declaring",
    "announced", "announces", "announce", "announcing",
    "stated", "states", "state", "stating",
    "remarked", "remarks", "remark", "remarking",
    "observed", "observes", "observe", "observing",
    "noted", "notes", "note", "noting",
    "added", "adds", "add", "adding",
    "continued", "continues", "continue", "continuing",
    "interrupted", "interrupts", "interrupt", "interrupting",
    "insisted", "insists", "insist", "insisting",
    "demanded", "demands", "demand", "demanding",
    "pleaded", "pleads", "plead", "pleading",
    "begged", "begs", "beg", "begging",
    "urged", "urges", "urge", "urging",
    "warned", "warns", "warn", "warning",
    "promised", "promises", "promise", "promising",
    "agreed", "agrees", "agree", "agreeing",
    "refused", "refuses", "refuse", "refusing",
    "denied", "denies", "deny", "denying",
    "admitted", "admits", "admit", "admitting",
    "confessed", "confesses", "confess", "confessing",
    "suggested", "suggests", "suggest", "suggesting",
    "proposed", "proposes", "propose", "proposing",
    "offered", "offers", "offer", "offering",
    "commanded", "commands", "command", "commanding",
    "ordered", "orders", "order", "ordering",
    "instructed", "instructs", "instruct", "instructing",
    "requested", "requests", "request", "requesting",
    "inquired", "inquires", "inquire", "inquiring",
    "questioned", "questions", "question", "questioning",
    "grumbled", "grumbles", "grumble", "grumbling",
    "stammered", "stammers", "stammer", "stammering",
    "stuttered", "stutters", "stutter", "stuttering",
    "whimpered", "whimpers", "whimper", "whimpering",
    "moaned", "moans", "moan", "moaning",
    "cheered", "cheers", "cheer", "cheering",
    "jeered", "jeers", "jeer", "jeering",
    "taunted", "taunts", "taunt", "taunting",
    "teased", "teases", "tease", "teasing",
    "joked", "jokes", "joke", "joking",
    "quipped", "quips", "quip", "quipping",
    "retorted", "retorts", "retort", "retorting",
})

ACTION_VERBS: FrozenSet[str] = frozenset({
    "nodded", "shook", "smiled", "frowned", "grimaced",
    "shrugged", "sighed", "laughed", "chuckled", "giggled",
    "scowled", "glared", "stared", "gaped", "blanched",
    "flinched", "winced", "cringed", "shuddered", "trembled",
    "paced", "strode", "stepped", "leaned", "turned",
    "rose", "stood", "sat", "collapsed", "slumped",
    "reached", "grabbed", "clutched", "clasped", "gripped",
    "wiped", "rubbed", "scratched", "folded", "crossed",
    "pushed", "pulled", "slammed", "tossed", "threw",
    "walked", "ran", "rushed", "hurried", "crept",
    "sneaked", "dashed", "fled", "retreated", "advanced",
    "drew", "pointed", "gestured", "waved", "beckoned",
    "looked", "watched", "observed", "noticed", "peeked",
    "closed", "opened", "locked", "unlocked", "sealed",
})

PRONOUNS_MALE: FrozenSet[str] = frozenset({
    "he", "him", "his", "himself",
})

PRONOUNS_FEMALE: FrozenSet[str] = frozenset({
    "she", "her", "hers", "herself",
})

PRONOUNS_NEUTRAL: FrozenSet[str] = frozenset({
    "they", "them", "their", "theirs", "themselves",
    "it", "its", "itself",
})

ALL_PRONOUNS: FrozenSet[str] = PRONOUNS_MALE | PRONOUNS_FEMALE | PRONOUNS_NEUTRAL

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet",
    "in", "on", "at", "to", "by", "of", "from", "with", "as", "into",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    "this", "that", "these", "those", "it", "its",
    "not", "no", "very", "just", "also", "then", "than", "too",
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor",
    "sir", "madam", "lord", "lady", "captain", "colonel",
    "rev", "reverend", "senator", "judge", "general",
    "uncle", "aunt", "cousin",
}

FEMALE_NAME_ENDINGS: FrozenSet[str] = frozenset({
    "a", "e", "i", "y", "ia", "ie", "ine", "ette",
    "ella", "elle", "ina", "ita", "ana", "ena",
    "ice", "ise", "yse", "is", "bel", "tle",
})

OPEN_QUOTES: FrozenSet[str] = frozenset({
    "\u201c",  # "
    "\u2018",  # '
    '"',
    "'",
    "\u00ab",  # «
})

CLOSE_QUOTES: FrozenSet[str] = frozenset({
    "\u201d",  # "
    "\u2019",  # '
    '"',
    "'",
    "\u00bb",  # »
})

ALL_QUOTE_CHARS: FrozenSet[str] = OPEN_QUOTES | CLOSE_QUOTES

SCENE_BREAK_RE: re.Pattern = re.compile(
    r"""
    ^\s*                                 # leading whitespace
    (?:                                  # --- option group ---
        \*{3,}                           #   *** or more
      | -{3,}                            #   --- or more
      | _{3,}                            #   ___ or more
      | ~{3,}                            #   ~~~ or more
      | ={3,}                            #   === or more
      | \#{2,}                           #   ## or more
      | \*                               #   single centered *
      | -{2,}                            #   -- or more
      | (?:\*\s*){3,}\*?                 #   * * * pattern
      | (?:-\s*){3,}-?                   #   - - - pattern
      | (?:\.\s*){4,}\.?                 #   . . . . pattern
    )
    \s*$                                 # trailing whitespace
    """,
    re.VERBOSE | re.MULTILINE,
)


# ===================================================================
# 2. SPEAKER ENGINE DATA STRUCTURES
# ===================================================================

class Confidence(Enum):
    """Confidence level for a speaker assignment.

    Ordering matters: EXPLICIT > PRONOUN > TURN > PROXIMITY > FALLBACK.
    """
    EXPLICIT = 5
    PRONOUN = 4
    TURN = 3
    PROXIMITY = 2
    FALLBACK = 1

    def __lt__(self, other: "Confidence") -> bool:
        return self.value < other.value

    def __le__(self, other: "Confidence") -> bool:
        return self.value <= other.value

    def __gt__(self, other: "Confidence") -> bool:
        return self.value > other.value

    def __ge__(self, other: "Confidence") -> bool:
        return self.value >= other.value


@dataclass(frozen=True)
class Character:
    """Immutable record for a known character in the text."""
    name: str
    gender: str = "unknown"           # "male", "female", "neutral", "unknown"
    mention_count: int = 0
    dialogue_count: int = 0
    aliases: Tuple[str, ...] = ()

    @property
    def is_named(self) -> bool:
        """True if the character has a proper name (not a pronoun)."""
        lower = self.name.lower()
        return lower not in ALL_PRONOUNS and not lower.startswith("unknown")

    def with_updates(
        self,
        *,
        mention_count: int | None = None,
        dialogue_count: int | None = None,
        gender: str | None = None,
        aliases: Tuple[str, ...] | None = None,
    ) -> "Character":
        return Character(
            name=self.name,
            gender=gender if gender is not None else self.gender,
            mention_count=mention_count if mention_count is not None else self.mention_count,
            dialogue_count=dialogue_count if dialogue_count is not None else self.dialogue_count,
            aliases=aliases if aliases is not None else self.aliases,
        )


@dataclass
class SpeakerAssignment:
    """Result of speaker detection for one text segment."""
    speaker: str
    confidence: Confidence
    raw_evidence: str
    is_dialogue: bool


@dataclass
class TextSegment:
    """A parsed segment of the input text — either dialogue or narrative."""
    text: str
    is_dialogue: bool
    is_narrative: bool
    quoted_text: str
    narrative_text: str
    paragraph_index: int
    sentence_index: int


# ===================================================================
# 3. CHARACTER REGISTRY
# ===================================================================

class CharacterRegistry:
    """Central registry for all characters discovered in the text.

    Manages name → character mapping, alias resolution, pronoun tracking,
    turn-taking state, and scene-level speaker sets.
    """

    def __init__(self) -> None:
        self._chars: Dict[str, Character] = {}
        self._alias_map: Dict[str, str] = {}          # alias → canonical name
        self._last_male: Optional[str] = None
        self._last_female: Optional[str] = None
        self._last_person: Optional[str] = None
        self._recent_speakers: List[str] = []          # most recent first
        self._scene_speakers: List[str] = []            # speakers in current scene

    # -- Properties -------------------------------------------------------

    @property
    def characters(self) -> Dict[str, Character]:
        return dict(self._chars)

    @property
    def last_speaker(self) -> Optional[str]:
        if self._recent_speakers:
            return self._recent_speakers[0]
        return None

    # -- Public API -------------------------------------------------------

    def get_sorted_speakers(self) -> List[str]:
        """Return character names sorted by dialogue count (desc), then mention count."""
        return [
            c.name for c in sorted(
                self._chars.values(),
                key=lambda c: (c.dialogue_count, c.mention_count),
                reverse=True,
            )
        ]

    def register(self, name: str, *, gender: str | None = None, is_dialogue: bool = False) -> Character:
        """Register a character or update an existing one."""
        canonical = self._resolve_alias(name) or name.strip()
        if not canonical:
            canonical = "Unknown"

        if canonical in self._chars:
            ch = self._chars[canonical]
            new_mentions = ch.mention_count + 1
            new_dialogues = ch.dialogue_count + (1 if is_dialogue else 0)
            new_gender = gender if gender is not None else ch.gender
            if new_gender == "unknown":
                inferred = self._infer_gender(canonical)
                if inferred != "unknown":
                    new_gender = inferred
            updated = ch.with_updates(
                mention_count=new_mentions,
                dialogue_count=new_dialogues,
                gender=new_gender,
            )
            self._chars[canonical] = updated
        else:
            resolved_gender = gender if gender else self._infer_gender(canonical)
            ch = Character(
                name=canonical,
                gender=resolved_gender,
                mention_count=1,
                dialogue_count=1 if is_dialogue else 0,
            )
            self._chars[canonical] = ch

        # Track by gender for pronoun resolution
        canonical_ch = self._chars[canonical]
        if canonical_ch.gender == "male":
            self._last_male = canonical
        elif canonical_ch.gender == "female":
            self._last_female = canonical
        self._last_person = canonical

        return self._chars[canonical]

    def update_gender(self, name: str, gender: str) -> None:
        """Force-update the gender of a character."""
        canonical = self._resolve_alias(name) or name.strip()
        if canonical in self._chars:
            ch = self._chars[canonical]
            self._chars[canonical] = ch.with_updates(gender=gender)
            if gender == "male":
                self._last_male = canonical
            elif gender == "female":
                self._last_female = canonical

    def resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Resolve a pronoun to the most likely character name."""
        plower = pronoun.lower()
        if plower in PRONOUNS_MALE:
            if self._last_male:
                return self._last_male
            # Fallback: most recent male-gendered character in scene
            for name in reversed(self._scene_speakers):
                ch = self._chars.get(name)
                if ch and ch.gender == "male":
                    return name
            return self._last_male
        if plower in PRONOUNS_FEMALE:
            if self._last_female:
                return self._last_female
            for name in reversed(self._scene_speakers):
                ch = self._chars.get(name)
                if ch and ch.gender == "female":
                    return name
            return self._last_female
        if plower in PRONOUNS_NEUTRAL:
            if self._last_person:
                return self._last_person
            if self._recent_speakers:
                return self._recent_speakers[0]
        return None

    def push_speaker(self, name: str) -> None:
        """Record the most recent speaker for turn-taking analysis."""
        canonical = self._resolve_alias(name) or name.strip()
        # Move to front if already in recent list
        if canonical in self._recent_speakers:
            self._recent_speakers.remove(canonical)
        self._recent_speakers.insert(0, canonical)
        # Keep bounded
        if len(self._recent_speakers) > 20:
            self._recent_speakers = self._recent_speakers[:20]
        # Also add to scene speakers
        if canonical not in self._scene_speakers:
            self._scene_speakers.append(canonical)

    def get_turn_taking_speaker(self, *, exclude: str | None = None) -> Optional[str]:
        """Return the most likely speaker based on turn-taking (alternation).

        In a two-person dialogue, the speaker alternates.  With more
        participants we fall back to the second-most-recent speaker.
        """
        candidates = self._recent_speakers[:]
        if exclude:
            canonical_exclude = self._resolve_alias(exclude) or exclude
            candidates = [c for c in candidates if c != canonical_exclude]
        if not candidates:
            return None
        # For 2-person dialogue, alternate
        if len(self._recent_speakers) >= 2 and exclude:
            return candidates[0]
        # For multi-party, return the second-most-recent
        if len(candidates) >= 2:
            return candidates[1] if candidates[0] == self.last_speaker else candidates[0]
        return candidates[0] if candidates else None

    def get_scene_speakers(self) -> List[str]:
        """Return list of speakers in the current scene."""
        return list(self._scene_speakers)

    def reset_scene(self) -> None:
        """Reset scene-level state (called at scene boundaries)."""
        self._scene_speakers.clear()
        # Keep recent_speakers for cross-scene continuity but trim
        if len(self._recent_speakers) > 5:
            self._recent_speakers = self._recent_speakers[:5]

    # -- Private helpers ---------------------------------------------------

    def _resolve_alias(self, name: str) -> Optional[str]:
        """Look up a name in the alias map."""
        stripped = name.strip()
        if stripped in self._alias_map:
            return self._alias_map[stripped]
        lower = stripped.lower()
        for alias, canonical in self._alias_map.items():
            if alias.lower() == lower:
                return canonical
        # Check direct character map
        if stripped in self._chars:
            return stripped
        return None

    @staticmethod
    def _infer_gender(name: str) -> str:
        """Infer gender from honorifics and name-ending patterns."""
        lower = name.lower().strip()

        # Honorific detection
        honorific_male = {"mr", "mr.", "sir", "lord", "king", "prince", "duke",
                          "captain", "colonel", "general", "dr", "dr.", "professor"}
        honorific_female = {"mrs", "mrs.", "ms", "ms.", "miss", "lady", "queen",
                            "princess", "duchess", "aunt", "sister", "mother",
                            "madam", "madame"}

        first_word = lower.split()[0].rstrip(".") if lower else ""
        if first_word in honorific_male or first_word + "." in honorific_male:
            return "male"
        if first_word in honorific_female or first_word + "." in honorific_female:
            return "female"

        # Name-ending heuristics (for first names only)
        first_name = lower.split()[0] if lower else ""
        if first_name:
            for ending in sorted(FEMALE_NAME_ENDINGS, key=len, reverse=True):
                if first_name.endswith(ending) and len(first_name) > len(ending):
                    return "female"
            # Common male endings / patterns
            male_endings = ("us", "os", "as", "is", "o", "n", "r", "d", "k")
            for ending in male_endings:
                if first_name.endswith(ending) and len(first_name) > 2:
                    # Not foolproof — only return male if we have some confidence
                    if ending in ("us", "os"):
                        return "male"

        return "unknown"

    def add_alias(self, alias: str, canonical: str) -> None:
        """Register an alias for a character."""
        alias_stripped = alias.strip()
        canonical_stripped = canonical.strip()
        if not alias_stripped or not canonical_stripped:
            return
        self._alias_map[alias_stripped] = canonical_stripped
        # If the character already exists, add alias to its record
        if canonical_stripped in self._chars:
            ch = self._chars[canonical_stripped]
            if alias_stripped not in ch.aliases:
                new_aliases = ch.aliases + (alias_stripped,)
                self._chars[canonical_stripped] = ch.with_updates(aliases=new_aliases)


# ===================================================================
# 4. DIALOGUE PARSER
# ===================================================================

class DialogueParser:
    """Parses raw text into structured TextSegment objects.

    Splits text by paragraphs, then within each paragraph separates
    dialogue (quoted) from narrative (unquoted) portions.
    """

    _QUOTE_RE: re.Pattern = re.compile(
        r'[\u201c"\u00ab][\s\S]*?[\u201d"\u00bb]'
        r'|[\u2018\'][\s\S]*?[\u2019\']'
    )

    _SPLIT_RE: re.Pattern = re.compile(
        r'([\u201c"\u00ab][\s\S]*?[\u201d"\u00bb]'
        r'|[\u2018\'][\s\S]*?[\u2019\'])'
    )

    @staticmethod
    def parse_paragraphs(text: str) -> List[TextSegment]:
        """Parse the full text into a flat list of TextSegments."""
        paragraphs = re.split(r'\n\s*\n', text)
        segments: List[TextSegment] = []
        sent_idx = 0

        for para_idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            # Check if the paragraph is entirely dialogue
            quoted_parts = DialogueParser._QUOTE_RE.findall(para)

            if not quoted_parts:
                # Pure narrative
                for sentence in DialogueParser._sentence_split_narrative(para):
                    segments.append(TextSegment(
                        text=sentence.strip(),
                        is_dialogue=False,
                        is_narrative=True,
                        quoted_text="",
                        narrative_text=sentence.strip(),
                        paragraph_index=para_idx,
                        sentence_index=sent_idx,
                    ))
                    sent_idx += 1
                continue

            # Mixed or pure dialogue — split on quote boundaries
            parts = DialogueParser._SPLIT_RE.split(para)
            for part in parts:
                part_stripped = part.strip()
                if not part_stripped:
                    continue
                is_dlg = bool(DialogueParser._QUOTE_RE.fullmatch(part_stripped))
                quoted = part_stripped if is_dlg else ""
                narrative = "" if is_dlg else part_stripped
                segments.append(TextSegment(
                    text=part_stripped,
                    is_dialogue=is_dlg,
                    is_narrative=not is_dlg,
                    quoted_text=quoted,
                    narrative_text=narrative,
                    paragraph_index=para_idx,
                    sentence_index=sent_idx,
                ))
                sent_idx += 1

        return segments

    @staticmethod
    def _split_mixed_segment(text: str) -> List[Tuple[str, bool]]:
        """Split a mixed narrative/dialogue segment into parts with is_dialogue flag."""
        result: List[Tuple[str, bool]] = []
        parts = DialogueParser._SPLIT_RE.split(text)
        for part in parts:
            s = part.strip()
            if not s:
                continue
            is_dlg = bool(DialogueParser._QUOTE_RE.fullmatch(s))
            result.append((s, is_dlg))
        return result

    @staticmethod
    def extract_quoted_text(text: str) -> str:
        """Extract the first quoted passage from text, stripping quote marks."""
        m = DialogueParser._QUOTE_RE.search(text)
        if m:
            return m.group(0)[1:-1].strip()
        return ""

    @staticmethod
    def count_quote_pairs(text: str) -> int:
        """Count the number of top-level quote pairs in the text."""
        return len(DialogueParser._QUOTE_RE.findall(text))

    @staticmethod
    def _sentence_split_narrative(text: str) -> List[str]:
        """Split narrative text into sentences (simplified)."""
        # Split on sentence-ending punctuation followed by space or end.
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p for p in parts if p.strip()]


# ===================================================================
# 5. SPEAKER DETECTION ENGINE
# ===================================================================

class SpeakerDetectionEngine:
    """Multi-pass speaker detection engine.

    Pipeline overview
    -----------------
    Pass 1  — Explicit attribution (regex cascade + optional spaCy NER)
    Pass 2  — Build / enrich character registry from pass-1 results
    Pass 3  — Pronoun coreference resolution
    Pass 4  — Turn-taking assignment
    Pass 5  — Proximity-based fallback
    Pass 6  — Scene-boundary reset & re-assignment
    Pass 7  — Dialogue-chain tracking (back-and-forth conversations)
    Pass 8  — Vocative detection (direct address in dialogue)
    Pass 9  — Possessive-voice detection ("Sarah's voice")
    Pass 10 — Final gap-fill and confidence normalization

    The public API consists of ``detect_speakers()`` and ``extract_speakers()``.
    """

    # -- Regex patterns for attribution -----------------------------------

    # "Name said" / "Name asked" etc.
    NAME_VERB: re.Pattern = re.compile(
        r'\b([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+'
        r'(said|says|asked|asks|replied|replies|answered|answers|'
        r'whispered|whispers|shouted|shouts|cried|cries|'
        r'exclaimed|exclaims|murmured|murmurs|muttered|mutters|'
        r'gasped|gasps|sighed|sighs|laughed|laughs|chuckled|chuckles|'
        r'snarled|snarls|growled|growls|hissed|hisses|'
        r'yelled|yells|screamed|screams|called|calls|'
        r'declared|declares|announced|announces|stated|states|'
        r'remarked|remarks|observed|observes|noted|notes|'
        r'added|adds|continued|continues|interrupted|interrupts|'
        r'insisted|insists|demanded|demands|pleaded|pleads|'
        r'begged|begs|urged|urges|warned|warns|promised|promises|'
        r'agreed|agrees|refused|refuses|denied|denies|'
        r'admitted|admits|confessed|confesses|suggested|suggests|'
        r'proposed|proposes|offered|offers|commanded|commands|'
        r'ordered|orders|instructed|instructs|requested|requests|'
        r'inquired|inquires|questioned|questions|grumbled|grumbles|'
        r'stammered|stammers|stuttered|stutters|whimpered|whimpers|'
        r'moaned|moans|cheered|cheers|jeered|jeers|taunted|taunts|'
        r'teased|teases|joked|jokes|quipped|quips|retorted|retorts|'
        r'barked|barks)\b',
        re.IGNORECASE,
    )

    # "said Name" / "asked Name"
    VERB_NAME: re.Pattern = re.compile(
        r'\b(said|asked|replied|answered|whispered|shouted|cried|'
        r'exclaimed|murmured|muttered|gasped|sighed|laughed|chuckled|'
        r'snarled|growled|hissed|yelled|screamed|called|'
        r'declared|announced|stated|remarked|observed|noted|'
        r'added|continued|interrupted|insisted|demanded|pleaded|'
        r'begged|urged|warned|promised|agreed|refused|denied|'
        r'admitted|confessed|suggested|proposed|offered|commanded|'
        r'ordered|instructed|requested|inquired|questioned|'
        r'grumbled|stammered|stuttered|whimpered|moaned|cheered|'
        r'jeered|taunted|teased|joked|quipped|retorted|barked)\s+'
        r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\b',
    )

    # "He said" / "She asked"
    PRONOUN_VERB: re.Pattern = re.compile(
        r'\b(He|She|They|It)\s+'
        r'(said|says|asked|asks|replied|replies|answered|answers|'
        r'whispered|whispers|shouted|shouts|cried|cries|'
        r'exclaimed|exclaims|murmured|murmurs|muttered|mutters|'
        r'gasped|gasps|sighed|sighs|laughed|laughs|chuckled|chuckles|'
        r'snarled|snarls|growled|growls|hissed|hisses|'
        r'yelled|yells|screamed|screams|called|calls|'
        r'declared|declares|announced|announces|stated|states|'
        r'remarked|remarks|observed|observes|noted|notes|'
        r'added|adds|continued|continues|interrupted|interrupts|'
        r'insisted|insists|demanded|demands|pleaded|pleads|'
        r'begged|begs|urged|urges|warned|warns|promised|promises|'
        r'agreed|agrees|refused|refuses|denied|denies|'
        r'admitted|admits|confessed|confesses|suggested|suggests|'
        r'proposed|proposes|offered|offers|commanded|commands|'
        r'ordered|orders|instructed|instructs|requested|requests|'
        r'inquired|inquires|questioned|questions|grumbled|grumbles|'
        r'stammered|stammers|stuttered|stutters|whimpered|whimpers|'
        r'moaned|moans|cheered|cheers|jeered|jeers|taunted|taunts|'
        r'teased|teases|joked|jokes|quipped|quips|retorted|retorts|'
        r'barked)\b',
    )

    # "said he" / "asked she"
    VERB_PRONOUN: re.Pattern = re.compile(
        r'\b(said|asked|replied|answered|whispered|shouted|cried|'
        r'exclaimed|murmured|muttered|gasped|sighed|laughed|chuckled|'
        r'snarled|growled|hissed|yelled|screamed|called|'
        r'declared|announced|stated|remarked|observed|noted|'
        r'added|continued|interrupted|insisted|demanded|pleaded|'
        r'begged|urged|warned|promised|agreed|refused|denied|'
        r'admitted|confessed|suggested|proposed|offered|commanded|'
        r'ordered|instructed|requested|inquired|questioned|'
        r'grumbled|stammered|stuttered|whimpered|moaned|cheered|'
        r'jeered|taunted|teased|joked|quipped|retorted|barked)\s+'
        r'(he|she|they|it|him|her|them)\b',
        re.IGNORECASE,
    )

    # Post-quote attribution: "…," Name said.
    POST_QUOTE_TAG: re.Pattern = re.compile(
        r'[\u201d"\u00bb]\s*,?\s*'
        r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+'
        r'(said|asks|asked|replied|answered|whispered|shouted|cried|'
        r'exclaimed|murmured|muttered|gasped|sighed|laughed|chuckled|'
        r'snarled|growled|hissed|yelled|screamed|called|'
        r'declared|announced|stated|remarked|observed|noted|'
        r'added|continued|interrupted|insisted|demanded|pleaded|'
        r'begged|urged|warned|promised|agreed|refused|denied|'
        r'admitted|confessed|suggested|proposed|offered|commanded|'
        r'ordered|instructed|requested|inquired|questioned|'
        r'grumbled|stammered|stuttered|whimpered|moaned|cheered|'
        r'jeered|taunted|teased|joked|quipped|retorted|barked)\b',
    )

    # Pre-quote attribution: Name said, "…"
    PRE_QUOTE_TAG: re.Pattern = re.compile(
        r'\b([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+'
        r'(said|asks|asked|replied|answered|whispered|shouted|cried|'
        r'exclaimed|murmured|muttered|gasped|sighed|laughed|chuckled|'
        r'snarled|growled|hissed|yelled|screamed|called|'
        r'declared|announced|stated|remarked|observed|noted|'
        r'added|continued|interrupted|insisted|demanded|pleaded|'
        r'begged|urged|warned|promised|agreed|refused|denied|'
        r'admitted|confessed|suggested|proposed|offered|commanded|'
        r'ordered|instructed|requested|inquired|questioned|'
        r'grumbled|stammered|stuttered|whimpered|moaned|cheered|'
        r'jeered|taunted|teased|joked|quipped|retorted|barked)\s*,?\s*'
        r'[\u201c"\u00ab]',
    )

    # Action subject: Name nodded / Name smiled
    ACTION_SUBJECT: re.Pattern = re.compile(
        r'\b([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+'
        r'(nodded|shook|smiled|frowned|grimaced|shrugged|sighed|'
        r'laughed|chuckled|giggled|scowled|glared|stared|gaped|'
        r'blanched|flinched|winced|cringed|shuddered|trembled|'
        r'paced|strode|stepped|leaned|turned|rose|stood|sat|'
        r'collapsed|slumped|reached|grabbed|clutched|clasped|'
        r'gripped|wiped|rubbed|scratched|folded|crossed|pushed|'
        r'pulled|slammed|tossed|threw|walked|ran|rushed|hurried|'
        r'crept|sneaked|dashed|fled|retreated|advanced|drew|'
        r'pointed|gestured|waved|beckoned|looked|watched|'
        r'observed|noticed|peeked|closed|opened|locked|unlocked|sealed)\b',
    )

    # Possessive voice: "Sarah's voice" / "his voice"
    POSSESSIVE_VOICE: re.Pattern = re.compile(
        r'\b([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*'
        r'|[Hh]e|[Ss]he|[Tt]hey|[Ii]t)'
        r'\'?s?\s+voice\b',
    )

    # Vocative: direct address inside dialogue
    VOCATIVE: re.Pattern = re.compile(
        r'[\u201c"\u00ab]\s*'
        r'(?:[A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*\s*[,.!?:]\s*)'
        r'|'
        r'(?:\s+[,.!?:]\s*'
        r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*))'
        r'[\u201d"\u00bb]',
    )

    # --------------------------------------------------------------------

    def __init__(self) -> None:
        self._registry = CharacterRegistry()
        self._scene_boundaries: List[int] = []

    @property
    def registry(self) -> CharacterRegistry:
        """Access the character registry after detection."""
        return self._registry

    # -- Public API -------------------------------------------------------

    def detect_speakers(self, text: str) -> List[SpeakerAssignment]:
        """Main entry point: run the full 10-pass pipeline and return assignments."""
        self._registry = CharacterRegistry()
        self._scene_boundaries = []

        segments = DialogueParser.parse_paragraphs(text)
        if not segments:
            return []

        # Detect scene boundaries
        self._scene_boundaries = self._detect_scene_boundaries(text)

        # Pass 1: explicit attribution
        assignments = self._pass1_explicit_attribution(segments, text)

        # Pass 2: build / enrich registry
        self._pass2_build_registry(assignments)

        # Pass 3-5: assign all speakers (pronoun, turn-taking, proximity)
        assignments = self._assign_all_speakers(assignments, segments)

        # Pass 6: scene-boundary adjustments
        assignments = self._pass6_scene_reset(assignments, segments)

        # Pass 7: dialogue-chain tracking
        assignments = self._pass7_dialogue_chains(assignments)

        # Pass 8: vocative detection
        assignments = self._pass8_vocative(assignments, segments)

        # Pass 9: possessive-voice detection
        assignments = self._pass9_possessive_voice(assignments, segments)

        # Pass 10: final gap-fill
        assignments = self._pass10_gap_fill(assignments, segments)

        return assignments

    def extract_speakers(self, text: str) -> List[str]:
        """Convenience: return just the ordered list of speaker names."""
        return [a.speaker for a in self.detect_speakers(text)]

    # -- Pass 1: Explicit Attribution ------------------------------------

    def _pass1_explicit_attribution(
        self, segments: List[TextSegment], full_text: str
    ) -> List[SpeakerAssignment]:
        """Find all explicitly-attributed speakers via regex cascade + spaCy."""
        assignments: List[SpeakerAssignment] = []
        prev_speaker: Optional[str] = None

        for seg in segments:
            speaker: Optional[str] = None
            evidence: str = ""
            confidence = Confidence.FALLBACK

            if seg.is_dialogue:
                speaker, evidence, confidence = self._find_explicit_speaker_dialogue(
                    seg, segments, prev_speaker
                )
            else:
                speaker, evidence, confidence = self._find_explicit_speaker_narrative(
                    seg, prev_speaker
                )

            # spaCy NER fallback for dialogue segments
            if speaker is None and seg.is_dialogue and _SPACY_OK:
                sp_name = self._spacy_extract_speaker(seg, segments)
                if sp_name:
                    speaker = sp_name
                    evidence = f"spaCy NER: {sp_name}"
                    confidence = Confidence.EXPLICIT

            if speaker:
                self._registry.push_speaker(speaker)
                prev_speaker = speaker

            assignments.append(SpeakerAssignment(
                speaker=speaker or "Unknown",
                confidence=confidence,
                raw_evidence=evidence,
                is_dialogue=seg.is_dialogue,
            ))

        return assignments

    def _find_explicit_speaker_dialogue(
        self,
        seg: TextSegment,
        all_segments: List[TextSegment],
        prev_speaker: Optional[str],
    ) -> Tuple[Optional[str], str, Confidence]:
        """Try to find an explicit speaker for a dialogue segment.

        Checks the current segment's narrative wrapper and adjacent segments.
        """
        # 1. Post-quote tag inside the same segment text
        m = self.POST_QUOTE_TAG.search(seg.text)
        if m:
            name = m.group(1)
            if not self._check_stopword(name):
                return name, f"post-quote tag: {m.group(0)}", Confidence.EXPLICIT

        # 2. Pre-quote tag inside the same segment text
        m = self.PRE_QUOTE_TAG.search(seg.text)
        if m:
            name = m.group(1)
            if not self._check_stopword(name):
                return name, f"pre-quote tag: {m.group(0)}", Confidence.EXPLICIT

        # 3. Name + speech verb in the same segment
        m = self.NAME_VERB.search(seg.text)
        if m:
            name = m.group(1)
            if not self._check_stopword(name):
                return name, f"name-verb: {m.group(0)}", Confidence.EXPLICIT

        # 4. Pronoun + speech verb
        m = self.PRONOUN_VERB.search(seg.text)
        if m:
            pronoun = m.group(1)
            resolved = self._registry.resolve_pronoun(pronoun)
            if resolved:
                return resolved, f"pronoun-verb: {m.group(0)} (→ {resolved})", Confidence.PRONOUN

        # 5. Check adjacent narrative segments
        idx = all_segments.index(seg) if seg in all_segments else -1
        if idx >= 0:
            # Check preceding narrative
            for offset in range(1, 4):
                if idx - offset >= 0:
                    neighbor = all_segments[idx - offset]
                    if neighbor.is_narrative:
                        name = self._find_character_in_narrative(neighbor.text)
                        if name:
                            return name, f"preceding narrative: {name}", Confidence.PROXIMITY
                        # Check for pronoun + speech verb
                        m = self.PRONOUN_VERB.search(neighbor.text)
                        if m:
                            resolved = self._registry.resolve_pronoun(m.group(1))
                            if resolved:
                                return resolved, f"preceding pronoun-verb: {m.group(0)} (→ {resolved})", Confidence.PRONOUN
                        break

            # Check following narrative
            for offset in range(1, 4):
                if idx + offset < len(all_segments):
                    neighbor = all_segments[idx + offset]
                    if neighbor.is_narrative:
                        m = self.POST_QUOTE_TAG.search(neighbor.text)
                        if m:
                            name = m.group(1)
                            if not self._check_stopword(name):
                                return name, f"following post-quote: {m.group(0)}", Confidence.EXPLICIT
                        m = self.NAME_VERB.search(neighbor.text)
                        if m:
                            name = m.group(1)
                            if not self._check_stopword(name):
                                return name, f"following name-verb: {m.group(0)}", Confidence.EXPLICIT
                        m = self.VERB_NAME.search(neighbor.text)
                        if m:
                            name = m.group(2)
                            if not self._check_stopword(name):
                                return name, f"following verb-name: {m.group(0)}", Confidence.EXPLICIT
                        m = self.VERB_PRONOUN.search(neighbor.text)
                        if m:
                            resolved = self._registry.resolve_pronoun(m.group(2))
                            if resolved:
                                return resolved, f"following verb-pronoun: {m.group(0)} (→ {resolved})", Confidence.PRONOUN
                        break

        return None, "", Confidence.FALLBACK

    def _find_explicit_speaker_narrative(
        self,
        seg: TextSegment,
        prev_speaker: Optional[str],
    ) -> Tuple[Optional[str], str, Confidence]:
        """Try to identify who is acting / speaking in a narrative segment."""
        # Name + speech verb
        m = self.NAME_VERB.search(seg.text)
        if m:
            name = m.group(1)
            if not self._check_stopword(name):
                return name, f"name-verb: {m.group(0)}", Confidence.EXPLICIT

        # Verb + name
        m = self.VERB_NAME.search(seg.text)
        if m:
            name = m.group(2)
            if not self._check_stopword(name):
                return name, f"verb-name: {m.group(0)}", Confidence.EXPLICIT

        # Pronoun + speech verb
        m = self.PRONOUN_VERB.search(seg.text)
        if m:
            resolved = self._registry.resolve_pronoun(m.group(1))
            if resolved:
                return resolved, f"pronoun-verb: {m.group(0)} (→ {resolved})", Confidence.PRONOUN

        # Verb + pronoun
        m = self.VERB_PRONOUN.search(seg.text)
        if m:
            resolved = self._registry.resolve_pronoun(m.group(2))
            if resolved:
                return resolved, f"verb-pronoun: {m.group(0)} (→ {resolved})", Confidence.PRONOUN

        # Action subject (for narrative, less confident)
        m = self.ACTION_SUBJECT.search(seg.text)
        if m:
            name = m.group(1)
            if not self._check_stopword(name):
                return name, f"action-subject: {m.group(0)}", Confidence.PROXIMITY

        return None, "", Confidence.FALLBACK

    # -- Pass 2: Build Registry ------------------------------------------

    def _pass2_build_registry(self, assignments: List[SpeakerAssignment]) -> None:
        """Enrich the character registry from pass-1 results and spaCy NER."""
        for a in assignments:
            if a.speaker and a.speaker != "Unknown":
                self._registry.register(a.speaker, is_dialogue=a.is_dialogue)

        # Also run spaCy over the full text if available to find all named entities
        if _SPACY_OK and _NLP is not None:
            # We do this lazily — it will be called in _spacy_register_all_characters
            pass

    # -- Passes 3-5: Assign All Speakers ---------------------------------

    def _assign_all_speakers(
        self,
        assignments: List[SpeakerAssignment],
        segments: List[TextSegment],
    ) -> List[SpeakerAssignment]:
        """Fill in speaker assignments using pronoun resolution, turn-taking, and proximity."""
        result = list(assignments)

        for i, (seg, assignment) in enumerate(zip(segments, result)):
            if assignment.speaker != "Unknown" and assignment.confidence >= Confidence.TURN:
                continue

            # Pass 3: Pronoun resolution
            if assignment.speaker == "Unknown" and seg.is_narrative:
                pronoun_speaker = self._resolve_narrative_pronoun(seg.text)
                if pronoun_speaker:
                    result[i] = SpeakerAssignment(
                        speaker=pronoun_speaker,
                        confidence=Confidence.PRONOUN,
                        raw_evidence=f"pronoun resolution in narrative",
                        is_dialogue=seg.is_dialogue,
                    )
                    self._registry.push_speaker(pronoun_speaker)
                    continue

            if assignment.speaker == "Unknown" and seg.is_dialogue:
                # Try pronoun in nearby narrative
                pronoun_speaker = self._resolve_dialogue_pronoun(i, segments, result)
                if pronoun_speaker:
                    result[i] = SpeakerAssignment(
                        speaker=pronoun_speaker,
                        confidence=Confidence.PRONOUN,
                        raw_evidence="pronoun resolution near dialogue",
                        is_dialogue=True,
                    )
                    self._registry.push_speaker(pronoun_speaker)
                    continue

            # Pass 4: Turn-taking
            if assignment.speaker == "Unknown" and seg.is_dialogue:
                turn_speaker = self._registry.get_turn_taking_speaker(
                    exclude=self._registry.last_speaker
                )
                if turn_speaker:
                    result[i] = SpeakerAssignment(
                        speaker=turn_speaker,
                        confidence=Confidence.TURN,
                        raw_evidence="turn-taking alternation",
                        is_dialogue=True,
                    )
                    self._registry.push_speaker(turn_speaker)
                    continue

            # Pass 5: Proximity
            if assignment.speaker == "Unknown":
                prox_speaker = self._find_proximate_speaker(i, segments, result)
                if prox_speaker:
                    confidence = Confidence.PROXIMITY
                    result[i] = SpeakerAssignment(
                        speaker=prox_speaker,
                        confidence=confidence,
                        raw_evidence="proximity fallback",
                        is_dialogue=seg.is_dialogue,
                    )
                    self._registry.push_speaker(prox_speaker)

        return result

    def _resolve_narrative_pronoun(self, text: str) -> Optional[str]:
        """Try to resolve a pronoun subject in narrative text."""
        # Check for subject pronouns
        for pronoun in ("He", "She", "They", "It"):
            if text.lstrip().startswith(pronoun + " "):
                resolved = self._registry.resolve_pronoun(pronoun)
                if resolved:
                    return resolved
        return None

    def _resolve_dialogue_pronoun(
        self,
        idx: int,
        segments: List[TextSegment],
        assignments: List[SpeakerAssignment],
    ) -> Optional[str]:
        """Look for pronoun hints near a dialogue segment."""
        # Check adjacent segments for pronoun+verb patterns
        for offset in range(-3, 4):
            if offset == 0:
                continue
            ni = idx + offset
            if 0 <= ni < len(segments):
                seg = segments[ni]
                if seg.is_narrative:
                    m = self.PRONOUN_VERB.search(seg.text)
                    if m:
                        resolved = self._registry.resolve_pronoun(m.group(1))
                        if resolved:
                            return resolved
        return None

    # -- Pass 6: Scene-Boundary Reset ------------------------------------

    def _pass6_scene_reset(
        self,
        assignments: List[SpeakerAssignment],
        segments: List[TextSegment],
    ) -> List[SpeakerAssignment]:
        """Apply scene-boundary logic — reset turn-taking at scene breaks."""
        if not self._scene_boundaries:
            return assignments

        result = list(assignments)
        current_para = -1

        for i, seg in enumerate(segments):
            if seg.paragraph_index != current_para:
                current_para = seg.paragraph_index
                if current_para in self._scene_boundaries:
                    self._registry.reset_scene()

            # After a scene reset, if we have a FALLBACK speaker, try to re-assign
            if result[i].confidence == Confidence.FALLBACK and result[i].is_dialogue:
                turn_speaker = self._registry.get_turn_taking_speaker()
                if turn_speaker:
                    result[i] = SpeakerAssignment(
                        speaker=turn_speaker,
                        confidence=Confidence.TURN,
                        raw_evidence="post-scene-reset turn-taking",
                        is_dialogue=True,
                    )

        return result

    # -- Pass 7: Dialogue-Chain Tracking ---------------------------------

    def _pass7_dialogue_chains(
        self, assignments: List[SpeakerAssignment]
    ) -> List[SpeakerAssignment]:
        """Track back-and-forth dialogue chains for better turn-taking."""
        result = list(assignments)

        # Identify runs of consecutive dialogue
        chain_start: Optional[int] = None
        for i, a in enumerate(result):
            if a.is_dialogue:
                if chain_start is None:
                    chain_start = i
            else:
                if chain_start is not None:
                    self._refine_dialogue_chain(result, chain_start, i)
                    chain_start = None
        # Handle chain that extends to end
        if chain_start is not None:
            self._refine_dialogue_chain(result, chain_start, len(result))

        return result

    def _refine_dialogue_chain(
        self,
        assignments: List[SpeakerAssignment],
        start: int,
        end: int,
    ) -> None:
        """Refine speaker assignments within a contiguous dialogue chain."""
        if end - start < 2:
            return

        # Collect the known speakers in the chain
        chain_speakers: List[Optional[str]] = []
        for i in range(start, end):
            if assignments[i].confidence >= Confidence.TURN:
                chain_speakers.append(assignments[i].speaker)
            else:
                chain_speakers.append(None)

        # If we have exactly 2 known speakers, fill gaps with alternation
        known = {s for s in chain_speakers if s is not None}
        if len(known) == 2:
            known_list = list(known)
            # Determine which speaker goes where based on the last explicitly known
            last_known_idx = None
            last_known_speaker = None
            for i, s in enumerate(chain_speakers):
                if s is not None:
                    last_known_idx = i
                    last_known_speaker = s

            # Fill forward from known positions
            prev_speaker = None
            for i in range(len(chain_speakers)):
                real_i = start + i
                if chain_speakers[i] is not None:
                    prev_speaker = chain_speakers[i]
                elif prev_speaker is not None and assignments[real_i].confidence < Confidence.TURN:
                    # Alternate
                    other = known_list[0] if prev_speaker == known_list[1] else known_list[1]
                    assignments[real_i] = SpeakerAssignment(
                        speaker=other,
                        confidence=Confidence.TURN,
                        raw_evidence="dialogue-chain alternation",
                        is_dialogue=True,
                    )
                    prev_speaker = other

    # -- Pass 8: Vocative Detection --------------------------------------

    def _pass8_vocative(
        self,
        assignments: List[SpeakerAssignment],
        segments: List[TextSegment],
    ) -> List[SpeakerAssignment]:
        """Detect vocative (direct address) patterns in dialogue."""
        result = list(assignments)

        for i, seg in enumerate(segments):
            if not seg.is_dialogue:
                continue
            if result[i].confidence >= Confidence.TURN:
                continue

            # Look for vocative pattern in quoted text
            quoted = seg.quoted_text
            if not quoted:
                continue

            # Try to find a name being directly addressed
            m = self.VOCATIVE.search(seg.text)
            if m:
                # The vocative identifies who is being spoken TO, not who is speaking.
                # But if we know the addressee, the speaker is likely someone else.
                addressee = m.group(1) if m.group(1) else None
                if addressee and not self._check_stopword(addressee):
                    self._registry.register(addressee)
                    # The speaker is probably not the addressee
                    turn_speaker = self._registry.get_turn_taking_speaker(exclude=addressee)
                    if turn_speaker:
                        result[i] = SpeakerAssignment(
                            speaker=turn_speaker,
                            confidence=Confidence.TURN,
                            raw_evidence=f"vocative exclusion of {addressee}",
                            is_dialogue=True,
                        )

        return result

    # -- Pass 9: Possessive-Voice Detection ------------------------------

    def _pass9_possessive_voice(
        self,
        assignments: List[SpeakerAssignment],
        segments: List[TextSegment],
    ) -> List[SpeakerAssignment]:
        """Detect "X's voice" / "his voice" patterns."""
        result = list(assignments)

        for i, seg in enumerate(segments):
            if result[i].confidence >= Confidence.TURN:
                continue

            m = self.POSSESSIVE_VOICE.search(seg.text)
            if m:
                possessor = m.group(1)
                # Resolve pronoun possessors
                if possessor.lower() in ALL_PRONOUNS:
                    resolved = self._registry.resolve_pronoun(possessor)
                    if resolved:
                        possessor = resolved
                elif not self._check_stopword(possessor):
                    self._registry.register(possessor)

                if possessor and possessor.lower() not in ALL_PRONOUNS:
                    result[i] = SpeakerAssignment(
                        speaker=possessor,
                        confidence=Confidence.PROXIMITY,
                        raw_evidence=f"possessive-voice: {m.group(0)}",
                        is_dialogue=seg.is_dialogue,
                    )

        return result

    # -- Pass 10: Final Gap-Fill -----------------------------------------

    def _pass10_gap_fill(
        self,
        assignments: List[SpeakerAssignment],
        segments: List[TextSegment],
    ) -> List[SpeakerAssignment]:
        """Final pass: fill any remaining Unknown or FALLBACK assignments."""
        result = list(assignments)

        # If spaCy is available, register all characters first
        if _SPACY_OK:
            full_text = " ".join(seg.text for seg in segments)
            self._spacy_register_all_characters(full_text)

        # Collect known speakers from the current scene
        scene_speakers = self._registry.get_scene_speakers()
        sorted_speakers = self._registry.get_sorted_speakers()

        for i, (seg, assignment) in enumerate(zip(segments, result)):
            if assignment.speaker != "Unknown":
                continue

            if seg.is_dialogue:
                # Try turn-taking one more time
                turn_speaker = self._registry.get_turn_taking_speaker(
                    exclude=self._registry.last_speaker
                )
                if turn_speaker:
                    result[i] = SpeakerAssignment(
                        speaker=turn_speaker,
                        confidence=Confidence.TURN,
                        raw_evidence="gap-fill turn-taking",
                        is_dialogue=True,
                    )
                    self._registry.push_speaker(turn_speaker)
                    continue

                # Try the most active speaker in the scene
                for ch in sorted_speakers:
                    if ch.is_named and ch.name != self._registry.last_speaker:
                        result[i] = SpeakerAssignment(
                            speaker=ch.name,
                            confidence=Confidence.FALLBACK,
                            raw_evidence="gap-fill most-active",
                            is_dialogue=True,
                        )
                        self._registry.push_speaker(ch.name)
                        break
                else:
                    # Last resort: use any named character
                    if sorted_speakers:
                        result[i] = SpeakerAssignment(
                            speaker=sorted_speakers[0].name,
                            confidence=Confidence.FALLBACK,
                            raw_evidence="gap-fill last-resort",
                            is_dialogue=True,
                        )
            else:
                # Narrative segment — try to find a character name
                name = self._find_character_in_narrative(seg.text)
                if name:
                    result[i] = SpeakerAssignment(
                        speaker=name,
                        confidence=Confidence.PROXIMITY,
                        raw_evidence="gap-fill character-in-narrative",
                        is_dialogue=False,
                    )
                elif sorted_speakers:
                    # Just use the most prominent character
                    result[i] = SpeakerAssignment(
                        speaker=sorted_speakers[0].name,
                        confidence=Confidence.FALLBACK,
                        raw_evidence="gap-fill narrative default",
                        is_dialogue=False,
                    )

        return result

    # -- Scene Boundary Detection ----------------------------------------

    def _detect_scene_boundaries(self, text: str) -> List[int]:
        """Detect scene breaks in the text, returning paragraph indices."""
        boundaries: List[int] = []
        paragraphs = re.split(r'\n\s*\n', text)

        for idx, para in enumerate(paragraphs):
            if SCENE_BREAK_RE.match(para.strip()):
                boundaries.append(idx)

        return boundaries

    # -- Helper Methods ---------------------------------------------------

    @staticmethod
    def _sentence_split(text: str) -> List[str]:
        """Split text into sentences (simplified, no NLTK dependency)."""
        # Handle common abbreviations
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Rev|Gen|Col|Capt|Lt|Sgt|Sr|Jr|St|vs|etc|approx|dept|est|govt|inc|ltd|dept|vol)\.\s',
                       lambda m: m.group(0).replace('.', '\x00'), text)
        # Split on sentence boundaries
        parts = re.split(r'(?<=[.!?])\s+', text)
        # Restore abbreviations
        return [p.replace('\x00', '.') for p in parts if p.strip()]

    @staticmethod
    def _check_stopword(name: str) -> bool:
        """Check if a detected 'name' is actually a stopword or speech verb."""
        lower = name.lower().strip()
        if lower in STOPWORDS:
            return True
        if lower in SPEECH_VERBS:
            return True
        if lower in ACTION_VERBS:
            return True
        # Check if it's just a single common word
        if len(lower) <= 2:
            return True
        return False

    def _find_proximate_speaker(
        self,
        idx: int,
        segments: List[TextSegment],
        assignments: List[SpeakerAssignment],
    ) -> Optional[str]:
        """Find the closest explicitly-attributed speaker in nearby segments."""
        # Search backwards
        for offset in range(1, 6):
            ni = idx - offset
            if 0 <= ni < len(assignments):
                a = assignments[ni]
                if a.speaker != "Unknown" and a.confidence >= Confidence.PRONOUN:
                    return a.speaker

        # Search forwards
        for offset in range(1, 6):
            ni = idx + offset
            if 0 <= ni < len(assignments):
                a = assignments[ni]
                if a.speaker != "Unknown" and a.confidence >= Confidence.PRONOUN:
                    return a.speaker

        return None

    def _find_character_in_narrative(self, text: str) -> Optional[str]:
        """Find a character name in narrative text."""
        # Try NAME_VERB pattern first
        m = self.NAME_VERB.search(text)
        if m:
            name = m.group(1)
            if not self._check_stopword(name):
                return name

        # Try ACTION_SUBJECT
        m = self.ACTION_SUBJECT.search(text)
        if m:
            name = m.group(1)
            if not self._check_stopword(name):
                return name

        # Try to find a capitalized name that's in our registry
        words = text.split()
        for i, word in enumerate(words):
            clean = word.strip(".,;:!?\"'\u201c\u201d")
            if clean and clean[0].isupper() and len(clean) > 2:
                if not self._check_stopword(clean):
                    resolved = self._registry._resolve_alias(clean)
                    if resolved:
                        return resolved
                    # Check if it's a multi-word name
                    if i + 1 < len(words):
                        next_clean = words[i + 1].strip(".,;:!?\"'\u201c\u201d")
                        if next_clean and next_clean[0].isupper():
                            full = f"{clean} {next_clean}"
                            if not self._check_stopword(full):
                                resolved = self._registry._resolve_alias(full)
                                if resolved:
                                    return resolved

        return None

    # -- spaCy Helpers ----------------------------------------------------

    @staticmethod
    def _spacy_extract_speaker(
        seg: TextSegment,
        all_segments: List[TextSegment],
    ) -> Optional[str]:
        """Use spaCy NER to find a speaker entity near a dialogue segment."""
        if not _SPACY_OK or _NLP is None:
            return None

        idx = all_segments.index(seg) if seg in all_segments else -1
        context_parts = [seg.text]
        if idx > 0:
            context_parts.insert(0, all_segments[idx - 1].text)
        if idx < len(all_segments) - 1:
            context_parts.append(all_segments[idx + 1].text)

        context = " ".join(context_parts)
        doc = _NLP(context)

        # Find PERSON entities that aren't in quotes
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                if name and not SpeakerDetectionEngine._check_stopword(name):
                    return name

        return None

    @staticmethod
    def _spacy_register_all_characters(text: str) -> None:
        """Use spaCy to find all PERSON entities and register them."""
        if not _SPACY_OK or _NLP is None:
            return

        # Process in chunks to avoid memory issues
        chunk_size = 5000
        for start in range(0, len(text), chunk_size):
            chunk = text[start:start + chunk_size]
            doc = _NLP(chunk)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip()
                    if name and not SpeakerDetectionEngine._check_stopword(name):
                        # The registry is accessed through the engine instance;
                        # we can't register here without a reference, so this is
                        # a no-op stub.  Real registration happens in _pass2_build_registry
                        # and _pass10_gap_fill via the engine's registry.
                        pass

    # -- Multi-word proper noun validation --------------------------------

    @staticmethod
    def _validate_proper_noun(name: str) -> bool:
        """Validate that a detected name is a plausible proper noun.

        Checks:
        - Not a stopword
        - Not a speech/action verb
        - Each word starts with a capital letter
        - Not all uppercase (likely an acronym or abbreviation)
        """
        if not name or not name.strip():
            return False

        name = name.strip()

        # Check stopwords
        if SpeakerDetectionEngine._check_stopword(name):
            return False

        # Check each component word
        words = name.split()
        for word in words:
            if not word:
                return False
            # Must start with uppercase
            if not word[0].isupper():
                return False
            # Must have at least one lowercase letter (not an acronym)
            if word.isupper() and len(word) > 1:
                return False

        return True

    # -- Confidence scoring helpers ---------------------------------------

    @staticmethod
    def _confidence_score(assignment: SpeakerAssignment) -> float:
        """Compute a numeric confidence score (0.0 – 1.0) for an assignment."""
        base = {
            Confidence.EXPLICIT: 0.95,
            Confidence.PRONOUN: 0.75,
            Confidence.TURN: 0.60,
            Confidence.PROXIMITY: 0.40,
            Confidence.FALLBACK: 0.15,
        }
        return base.get(assignment.confidence, 0.1)


# ===================================================================
# 6. CONVENIENCE FUNCTIONS (Speaker Engine)
# ===================================================================

def detect_speakers(text: str) -> List[str]:
    """Quick convenience: return unique sorted speaker names for the text."""
    engine = SpeakerDetectionEngine()
    engine.detect_speakers(text)
    return engine.registry.get_sorted_speakers()


def analyze_text(text: str) -> Tuple[List[SpeakerAssignment], CharacterRegistry]:
    """Full analysis: return assignments and the populated character registry."""
    engine = SpeakerDetectionEngine()
    assignments = engine.detect_speakers(text)
    return assignments, engine.registry


def get_speaker_assignments(text: str) -> List[Tuple[str, str, str]]:
    """Return a list of (speaker, confidence_name, evidence) tuples."""
    engine = SpeakerDetectionEngine()
    assignments = engine.detect_speakers(text)
    return [
        (a.speaker, a.confidence.name, a.raw_evidence)
        for a in assignments
    ]


# ===================================================================
# 7. TEXT PROCESSOR — DATACLASSES & CONSTANTS
# ===================================================================

@dataclass(frozen=True, slots=True)
class Chapter:
    """An immutable chapter detected in source text."""
    index: int
    title: str
    text: str
    word_count: int = 0

    def __post_init__(self):
        if self.word_count == 0 and self.text:
            object.__setattr__(self, "word_count", len(self.text.split()))


@dataclass(frozen=True, slots=True)
class TextStats:
    """Summary statistics for a text source."""
    char_count: int
    word_count: int
    estimated_minutes: float
    chapter_count: int
    speakers_detected: int
    # Extended with reading-level analysis
    flesch_kincaid_grade: float = 0.0
    flesch_reading_ease: float = 0.0
    complexity_score: float = 0.0

    @property
    def estimated_str(self) -> str:
        m = int(self.estimated_minutes)
        if m < 1:
            return "<1 min"
        h, r = divmod(m, 60)
        return f"{h}h {r}m" if h else f"{m}m"


@dataclass(frozen=True, slots=True)
class DialogueChunk:
    """
    A chunk of text containing one or more segments.
    Designed for SSML multi-voice synthesis.
    """
    text: str                                     # Full combined text (for fallback backends)
    dominant_speaker: str                         # The speaker with the most text in this chunk
    segments: Tuple[Tuple[str, str], ...]         # ((text, speaker), (text, speaker), ...)


@dataclass(frozen=True, slots=True)
class ReadingLevelStats:
    """Detailed reading-level analysis results."""
    flesch_kincaid_grade: float
    flesch_reading_ease: float
    avg_sentence_length: float
    avg_syllables_per_word: float
    complexity_score: float       # 0-100, higher = more complex
    word_count: int
    sentence_count: int
    estimated_listening_minutes: float


WPM_ESTIMATE = 155.0

CHAPTER_RE = re.compile(
    r"^(chapter\s+\d+|part\s+\d+|section\s+\d+|"
    r"prologue|epilogue|introduction|preface|conclusion|"
    r"book\s+\d+|act\s+\d+|scene\s+\d+)\b",
    re.MULTILINE | re.IGNORECASE,
)

# Improved scene-break chapter detection pattern
SCENE_BREAK_CHAPTER_RE = re.compile(
    r"""
    ^\s*
    (?:                                  # --- scene break markers ---
        \*{3,}                           #   *** or more
      | -{3,}                            #   --- or more
      | _{3,}                            #   ___ or more
      | ~{3,}                            #   ~~~ or more
      | ={3,}                            #   === or more
      | \#{2,}                           #   ## or more (markdown headings)
      | (?:\*\s*){3,}\*?                 #   * * * pattern
      | (?:-\s*){3,}-?                   #   - - - pattern
      | (?:\.\s*){4,}\.?                 #   . . . . pattern
    )
    \s*$
    """,
    re.VERBOSE | re.MULTILINE,
)

# ── Syllable counting heuristics ────────────────────────────────────────

_SPECIAL_SYLLABLES: Dict[str, int] = {
    # Common irregulars
    "the": 1, "be": 1, "been": 1, "every": 3, "different": 3,
    "business": 2, "probably": 3, "family": 3, "finally": 3,
    "general": 3, "beautiful": 3, "certain": 2, "realize": 3,
    "realized": 3, "genuine": 3, "figure": 2, "nature": 2,
    "measure": 2, "treasure": 2, "failure": 3, "creature": 2,
    "future": 2, "picture": 2, "literature": 4, "temperature": 4,
    "comfortable": 4, "vegetable": 4, "chocolate": 3, "interesting": 4,
    "necessary": 4, "experience": 4, "remember": 3, "powerful": 3,
    "dangerous": 3, "mysterious": 4, "happiness": 3, "everything": 4,
    "something": 2, "nothing": 2, "whatever": 3, "whenever": 3,
    "wherever": 3, "however": 3, "together": 3, "another": 3,
    "either": 2, "neither": 2, "whether": 2, "slightly": 2,
    "quietly": 2, "exactly": 3, "suddenly": 3, "merely": 2,
    "nearly": 2, "clearly": 2, "really": 2, "simply": 2,
    "terribly": 3, "incredible": 4, "invisible": 4, "impossible": 4,
    "uncomfortable": 5, "unbelievable": 5, "disappointing": 4,
    "extraordinary": 6, "immediately": 5, "particularly": 5,
    "unfortunately": 5, "significantly": 5, "consequently": 4,
    "occasionally": 5, "fundamentally": 5, "approximately": 5,
    "simultaneously": 6, "extraordinarily": 6,
}

# Common English words that are unusual for TTS and may need pronunciation hints
_UNUSUAL_WORDS: Dict[str, str] = {
    # Fantasy / literary terms
    "chthonic": "THON-ik",
    "cynosure": "SY-nuh-shoor",
    "demese": "dih-MEEZ",
    "ephemeral": "eh-FEM-er-ul",
    "epiphany": "eh-PIF-uh-nee",
    "euphemism": "YOO-fuh-miz-um",
    "hubris": "HYOO-bris",
    "idyll": "EYE-dul",
    "inchoate": "in-KOH-it",
    "lacuna": "luh-KOO-nuh",
    "legerdemain": "LEJ-er-duh-mayn",
    "metempsychosis": "meh-tem-SY-koh-sis",
    "mimesis": "mih-MEE-sis",
    "nadir": "NAY-der",
    "nonpareil": "non-puh-REL",
    "penumbra": "peh-NUM-bruh",
    "peripatetic": "per-ih-puh-TET-ik",
    "picaresque": "pik-uh-RESK",
    "prescient": "PRESH-unt",
    "pusillanimous": "pyoo-sih-LAN-ih-mus",
    "quotidian": "kwoh-TID-ee-un",
    "recrudescent": "ree-kroo-DES-unt",
    "sanguine": "SANG-gwin",
    "selvedge": "SEL-vidj",
    "sublime": "suh-BLYM",
    "supernal": "soo-PER-nul",
    "sycophant": "SIK-uh-fant",
    "talisman": "TAL-iz-man",
    "tergiversation": "ter-jih-ver-SAY-shun",
    "verisimilitude": "ver-ih-sih-MIL-ih-tood",
    # Names commonly mangled by TTS
    "ng": "eng",
    "nguyen": "nwen",
    "saoirse": "SEER-shuh",
    "siobhan": "shi-VAWN",
    "xerxes": "ZERK-seez",
    # Place names
    "gloucester": "GLO-ster",
    "leicester": "LES-ter",
    "worcester": "WUSS-ter",
    "edinburgh": "ED-in-bur-uh",
    "reykjavik": "RAYK-yuh-veek",
}


# ===================================================================
# 8. TEXT PROCESSOR CLASS
# ===================================================================

class TextProcessor:
    """Stateless text processing — all methods are static / class methods.

    Multi-speaker detection is handled by the inline
    SpeakerDetectionEngine (10-pass pipeline).
    """

    # ── Reading Level Analysis ──────────────────────────────────────

    @staticmethod
    def reading_level(text: str) -> ReadingLevelStats:
        """Compute reading-level analysis using Flesch-Kincaid formulas.

        Returns a ReadingLevelStats dataclass with grade level, reading ease,
        complexity score (0–100), word count, sentence count, and estimated
        listening time.
        """
        if not text or not text.strip():
            return ReadingLevelStats(
                flesch_kincaid_grade=0.0,
                flesch_reading_ease=100.0,
                avg_sentence_length=0.0,
                avg_syllables_per_word=0.0,
                complexity_score=0.0,
                word_count=0,
                sentence_count=0,
                estimated_listening_minutes=0.0,
            )

        words = text.split()
        word_count = len(words)
        if word_count == 0:
            return ReadingLevelStats(
                flesch_kincaid_grade=0.0,
                flesch_reading_ease=100.0,
                avg_sentence_length=0.0,
                avg_syllables_per_word=0.0,
                complexity_score=0.0,
                word_count=0,
                sentence_count=0,
                estimated_listening_minutes=0.0,
            )

        # Count sentences
        sentence_count = len(re.findall(r'[.!?]+', text))
        if sentence_count == 0:
            sentence_count = 1

        # Count total syllables
        total_syllables = sum(TextProcessor._count_syllables(w) for w in words)

        avg_sentence_length = word_count / sentence_count
        avg_syllables_per_word = total_syllables / word_count

        # Flesch Reading Ease (higher = easier)
        # 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        flesch_reading_ease = (
            206.835
            - 1.015 * avg_sentence_length
            - 84.6 * avg_syllables_per_word
        )
        # Clamp to 0–100
        flesch_reading_ease = max(0.0, min(100.0, flesch_reading_ease))

        # Flesch-Kincaid Grade Level
        # 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        flesch_kincaid_grade = (
            0.39 * avg_sentence_length
            + 11.8 * avg_syllables_per_word
            - 15.59
        )
        flesch_kincaid_grade = max(0.0, flesch_kincaid_grade)

        # Complexity score: 0–100, where 100 is most complex
        # Combines grade level and sentence length into a normalized score
        complexity_score = min(100.0, max(0.0, (
            (flesch_kincaid_grade / 18.0) * 50.0  # grade contributes up to 50
            + (avg_sentence_length / 40.0) * 25.0  # sentence length up to 25
            + (avg_syllables_per_word / 3.0) * 25.0  # syllable density up to 25
        )))

        estimated_listening_minutes = word_count / WPM_ESTIMATE

        return ReadingLevelStats(
            flesch_kincaid_grade=round(flesch_kincaid_grade, 2),
            flesch_reading_ease=round(flesch_reading_ease, 2),
            avg_sentence_length=round(avg_sentence_length, 2),
            avg_syllables_per_word=round(avg_syllables_per_word, 2),
            complexity_score=round(complexity_score, 2),
            word_count=word_count,
            sentence_count=sentence_count,
            estimated_listening_minutes=round(estimated_listening_minutes, 2),
        )

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Estimate syllable count for a single word using heuristic rules.

        Uses a special-case lookup table for common irregulars, then
        falls back to a standard vowel-counting heuristic.
        """
        word_lower = word.lower().strip(".,;:!?\"'\u201c\u201d")
        if not word_lower:
            return 0

        # Check special cases
        if word_lower in _SPECIAL_SYLLABLES:
            return _SPECIAL_SYLLABLES[word_lower]

        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_was_vowel = False

        for char in word_lower:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        # Subtract for silent 'e' at end
        if word_lower.endswith("e") and count > 1:
            # But not for -le endings (like "table")
            if len(word_lower) >= 2 and word_lower[-2] == "l":
                pass  # keep count as-is
            elif word_lower.endswith("le") and len(word_lower) > 2 and word_lower[-3] not in vowels:
                pass  # -cle, -ble, -dle etc. — the e is not silent
            else:
                count -= 1

        # Handle -ed endings where the 'e' is not pronounced
        if word_lower.endswith("ed") and count > 1:
            if len(word_lower) >= 3 and word_lower[-3] not in ("t", "d"):
                count -= 1

        return max(count, 1)

    # ── Pronunciation Auto-Suggest ──────────────────────────────────

    @staticmethod
    def suggest_pronunciations(text: str) -> Dict[str, str]:
        """Detect unusual words in text and suggest phonetic pronunciations.

        Scans the text for words that are likely to be mispronounced by
        TTS engines and returns a dictionary mapping each unusual word
        to a suggested phonetic spelling.

        Detection criteria:
        - Words in the built-in unusual-words dictionary
        - Words with low vowel-to-consonant ratio (possible foreign origin)
        - Words with unusual consonant clusters
        - Long words not in a common English frequency list
        """
        if not text or not text.strip():
            return {}

        suggestions: Dict[str, str] = {}

        # Extract unique words
        words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
        unique_words = set(words)

        # 1. Check built-in unusual-words dictionary
        for word in unique_words:
            if word in _UNUSUAL_WORDS:
                suggestions[word] = _UNUSUAL_WORDS[word]

        # 2. Detect unusual consonant clusters and patterns
        unusual_patterns = [
            (r"\b\w*[bcdfghjklmnpqrstvwxyz]{4,}\w*\b", "consonant cluster"),
            (r"\b\w*phth\w*\b", "phth cluster"),
            (r"\b\w*mn\w*\b", "mn cluster"),
            (r"\b\w*pt\w*\b", "pt cluster"),  # e.g., "ptarmigan"
            (r"\b\w*ae\w*\b", "ae ligature"),  # Latin-derived
            (r"\b\w*oe\w*\b", "oe ligature"),  # Latin-derived
        ]

        for pattern, reason in unusual_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                word = match.group(0).lower()
                if word not in suggestions and len(word) > 4:
                    # Generate a simple phonetic suggestion
                    phonetic = TextProcessor._generate_phonetic(word)
                    if phonetic != word:
                        suggestions[word] = phonetic

        # 3. Detect very long words that might need help
        for word in unique_words:
            if len(word) > 12 and word not in suggestions:
                phonetic = TextProcessor._generate_phonetic(word)
                if phonetic != word:
                    suggestions[word] = phonetic

        # 4. Words with apostrophes in unusual positions
        for word in unique_words:
            if "'" in word and word not in suggestions:
                # Check if it's a possessive or contraction (common, skip)
                if word.endswith("'s") or word.endswith("'t") or word.endswith("'ll"):
                    continue
                if word.endswith("'ve") or word.endswith("'re") or word.endswith("'d"):
                    continue
                if word in ("i'm", "you're", "they're", "we're", "it's", "that's"):
                    continue
                # Unusual apostrophe usage — might be a foreign word
                phonetic = TextProcessor._generate_phonetic(word)
                if phonetic != word:
                    suggestions[word] = phonetic

        return suggestions

    @staticmethod
    def _generate_phonetic(word: str) -> str:
        """Generate a simplified phonetic spelling for a word.

        Uses basic English pronunciation rules to produce a more
        phonetically-regular version of the word.
        """
        if not word:
            return word

        result = word.lower()

        # Common pronunciation normalizations
        replacements = [
            # Digraphs and special sounds
            ("ph", "f"),
            ("ough", "oh"),
            ("augh", "aw"),
            ("eigh", "ay"),
            ("igh", "eye"),
            # Silent letter patterns
            ("kn", "n"),
            ("gn", "n"),
            ("wr", "r"),
            ("ps", "s"),
            ("pn", "n"),
            ("pt", "t"),
            # Vowel combinations
            ("ae", "ee"),
            ("oe", "ee"),
            ("ie", "ee"),
            ("ei", "ay"),
            ("ou", "ow"),
            ("au", "aw"),
            ("aw", "aw"),
            # Consonant sounds
            ("ch", "k"),  # for Greek-origin words; not always right
            ("qu", "kw"),
            ("que", "k"),
        ]

        for old, new in replacements:
            # Only apply if the old pattern exists and replacement differs
            if old in result and old != new:
                result = result.replace(old, new)

        return result if result != word.lower() else word

    # ── Scene Break Detection ───────────────────────────────────────

    @staticmethod
    def detect_scene_breaks(text: str) -> List[int]:
        """Detect scene breaks in text, returning line numbers.

        Detects common scene-break markers:
        - ***  or  * * *  (asterisk rows)
        - ---  or  - - -  (em-dash rows)
        - ___  (underscore rows)
        - ###  (hash rows, markdown headings)
        - ===  (equals rows)
        - ~~~  (tilde rows)
        - . . . .  (dot rows)

        Returns a list of line numbers (0-indexed) where scene breaks occur.
        """
        if not text:
            return []

        breaks: List[int] = []
        for i, line in enumerate(text.splitlines()):
            if SCENE_BREAK_RE.match(line):
                breaks.append(i)

        return breaks

    # ── Compute Stats ───────────────────────────────────────────────

    @staticmethod
    def compute_stats(
        source: str,
        is_raw_text: bool = False,
    ) -> TextStats:
        full = TextProcessor._collect_text(source, is_raw_text)
        chars = len(full)
        words = len(full.split()) if full else 0
        est_min = words / WPM_ESTIMATE if words else 0
        chapters = TextProcessor.detect_chapters(source, is_raw_text)
        speakers = TextProcessor.extract_speakers(source, is_raw_text)

        # Compute reading-level analysis
        rl = TextProcessor.reading_level(full)

        return TextStats(
            char_count=chars,
            word_count=words,
            estimated_minutes=est_min,
            chapter_count=len(chapters),
            speakers_detected=len(speakers),
            flesch_kincaid_grade=rl.flesch_kincaid_grade,
            flesch_reading_ease=rl.flesch_reading_ease,
            complexity_score=rl.complexity_score,
        )

    @staticmethod
    def estimate_disk_mb(word_count: int, bitrate: str = "192k", output_format: str = "mp3") -> float:
        """Estimate output file size in MB based on word count and bitrate."""
        minutes = word_count / WPM_ESTIMATE if word_count else 0
        try:
            kbps = int(bitrate.replace("k", ""))
        except (ValueError, AttributeError):
            kbps = 192

        if output_format in ("wav", "both"):
            mb = minutes * 10.0
        elif output_format == "m4b":
            mb = minutes * (128 / 8 / 1024) * 60
        else:
            mb = minutes * (kbps / 8 / 1024) * 60

        return max(mb, 1.0)

    # ── Speaker Detection (delegated to inline engine) ──────────────

    @staticmethod
    def extract_speakers(
        source: str, is_raw_text: bool = False
    ) -> List[str]:
        """Extract all character names using the multi-pass SpeakerDetectionEngine.

        The engine runs 10 passes:
          1. Explicit attribution (regex + spaCy NER)
          2. Character registry construction with gender inference
          3. Pronoun coreference resolution ("he said" → character name)
          4. Dialogue turn-taking heuristic
          5. Context-window proximity resolution
          6. Scene boundary detection & state reset
          7. Dialogue-chain tracking
          8. Vocative address detection
          9. Possessive-voice pattern matching
         10. Final gap-fill & confidence normalization
        """
        full = TextProcessor._collect_text(source, is_raw_text)
        if not full.strip():
            return []
        return detect_speakers(full)

    @staticmethod
    def extract_speakers_detailed(
        source: str, is_raw_text: bool = False
    ) -> Tuple[List[str], CharacterRegistry]:
        """Extract speakers and return the full character registry with gender/mentions.

        Returns (speaker_names, registry) where registry contains Character
        objects with gender, mention_count, dialogue_count, and aliases.
        """
        full = TextProcessor._collect_text(source, is_raw_text)
        if not full.strip():
            return [], CharacterRegistry()

        _, registry = analyze_text(full)
        return registry.get_sorted_speakers(), registry

    # ── Chapter Detection (improved with scene breaks) ──────────────

    @staticmethod
    def detect_chapters(source: str, is_raw_text: bool = False) -> List[Chapter]:
        """Detect chapters and scene breaks in source text.

        Splits on:
        - Traditional chapter headings: "Chapter X", "Part Y", "Book Z", etc.
        - Scene break markers: ***, ###, ---, ===, ~~~, * * *, etc.

        Scene breaks that appear between chapters are treated as
        sub-sections.  Scene breaks at the top level (when no chapter
        headings are found) are promoted to chapter boundaries.
        """
        full = TextProcessor._collect_text(source, is_raw_text)
        if not full.strip():
            return []

        # First, try traditional chapter heading detection
        splits = CHAPTER_RE.split(full)
        if len(splits) <= 1:
            # No chapter headings found — try scene-break splitting
            return TextProcessor._chapters_from_scene_breaks(full)

        chapters: List[Chapter] = []
        i = 0
        idx = 0
        while i < len(splits):
            chunk = splits[i].strip()
            if not chunk:
                i += 1
                continue
            if CHAPTER_RE.match(chunk):
                title = chunk.strip()
                body = splits[i + 1].strip() if i + 1 < len(splits) else ""
                # Check for scene breaks within the chapter body
                sub_chapters = TextProcessor._split_by_scene_breaks(body, idx, title)
                if sub_chapters:
                    chapters.extend(sub_chapters)
                    idx += len(sub_chapters)
                else:
                    chapters.append(Chapter(index=idx, title=title, text=body))
                    idx += 1
                i += 2
            else:
                chapters.insert(0, Chapter(index=0, title="Introduction", text=chunk))
                for ci, c in enumerate(chapters):
                    object.__setattr__(c, "index", ci)
                idx = len(chapters)
                i += 1
        return chapters or [Chapter(index=0, title="Full Text", text=full.strip())]

    @staticmethod
    def _chapters_from_scene_breaks(full: str) -> List[Chapter]:
        """Split text into chapters based solely on scene-break markers.

        Used when no traditional chapter headings are found.
        """
        parts = SCENE_BREAK_CHAPTER_RE.split(full)
        if len(parts) <= 1:
            return [Chapter(index=0, title="Full Text", text=full.strip())]

        chapters: List[Chapter] = []
        idx = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Try to use the first non-empty line as a title hint
            first_line = part.split("\n", 1)[0].strip()
            if first_line and len(first_line) < 60:
                title = first_line
            else:
                title = f"Section {idx + 1}"
            chapters.append(Chapter(index=idx, title=title, text=part))
            idx += 1

        return chapters or [Chapter(index=0, title="Full Text", text=full.strip())]

    @staticmethod
    def _split_by_scene_breaks(body: str, start_idx: int, parent_title: str) -> List[Chapter]:
        """Split a chapter body by scene breaks into sub-chapters.

        Returns an empty list if there are no scene breaks (the chapter
        should be kept as a single unit).
        """
        parts = SCENE_BREAK_CHAPTER_RE.split(body)
        if len(parts) <= 1:
            return []

        chapters: List[Chapter] = []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            title = f"{parent_title} — Section {i + 1}" if len(parts) > 1 else parent_title
            chapters.append(Chapter(index=start_idx + i, title=title, text=part))

        return chapters

    # ── Metadata Extraction ─────────────────────────────────────────

    @staticmethod
    def extract_metadata(source: str, is_raw_text: bool = False) -> Dict[str, str]:
        """Extract book title and author from file metadata (EPUB, PDF, DOCX)."""
        meta: Dict[str, str] = {"title": "", "author": ""}

        if is_raw_text:
            first_line = source.strip().split("\n")[0].strip()
            if first_line and len(first_line) < 80:
                meta["title"] = first_line
            return meta

        if not Path(source).exists():
            return meta

        ext = Path(source).suffix.lower()

        try:
            if ext == '.epub' and EPUB_OK:
                book = epub.read_epub(source)
                metadata = book.get_metadata('DC', 'title')
                if metadata:
                    meta["title"] = metadata[0][0]
                metadata = book.get_metadata('DC', 'creator')
                if metadata:
                    meta["author"] = metadata[0][0]

            elif ext == '.pdf' and PDF_OK:
                reader = PdfReader(source)
                info = reader.metadata
                if info:
                    meta["title"] = info.get("/Title", "")
                    meta["author"] = info.get("/Author", "")

            elif ext == '.docx' and DOCX_OK:
                doc = Document(source)
                props = doc.core_properties
                meta["title"] = props.title or ""
                meta["author"] = props.author or ""

        except Exception:
            pass

        if not meta["title"]:
            meta["title"] = Path(source).stem.replace("_", " ").replace("-", " ").title()

        return meta

    # ── Pronunciation ───────────────────────────────────────────────

    @staticmethod
    def apply_pronunciation(text: str, pronunciation_dict: Dict[str, str]) -> str:
        """Apply pronunciation dictionary replacements to text."""
        if not pronunciation_dict:
            return text

        for source, replacement in pronunciation_dict.items():
            if not source or not replacement:
                continue
            pattern = re.compile(r'\b' + re.escape(source) + r'\b', re.IGNORECASE)
            text = pattern.sub(replacement, text)

        return text

    # ── Chunking ────────────────────────────────────────────────────

    @staticmethod
    def chunk_generator(
        source: str,
        max_chars: int = 500,
        is_raw_text: bool = False,
        multi_speaker: bool = False,
        pronunciation_dict: Optional[Dict[str, str]] = None,
    ) -> Generator[DialogueChunk, None, None]:
        """Yields DialogueChunk objects containing text and speaker segments.

        When multi_speaker is True, the professional SpeakerDetectionEngine
        is used for multi-pass speaker assignment with pronoun resolution,
        turn-taking heuristics, and confidence scoring.
        """
        full = TextProcessor._collect_text(source, is_raw_text)

        if pronunciation_dict:
            full = TextProcessor.apply_pronunciation(full, pronunciation_dict)

        if not multi_speaker:
            yield from TextProcessor._chunk_single_speaker(full, max_chars)
        else:
            yield from TextProcessor._chunk_multi_speaker(full, max_chars)

    # ── Single-Speaker Chunking ─────────────────────────────────────

    @staticmethod
    def _chunk_single_speaker(
        text: str, max_chars: int
    ) -> Generator[DialogueChunk, None, None]:
        """Chunk text for single-voice (Narrator) synthesis."""
        sentence_end = re.compile(r'(?<=[.!?。！？])\s+')
        buffer = ""
        for clean in TextProcessor._clean_stream([text]):
            buffer += clean
            while len(buffer) > max_chars:
                m = sentence_end.search(buffer, 0, max_chars + 80)
                if m:
                    piece = buffer[:m.end()].strip()
                    if piece:
                        yield DialogueChunk(piece, "Narrator", ((piece, "Narrator"),))
                    buffer = buffer[m.end():].strip()
                else:
                    piece = buffer[:max_chars].strip()
                    if piece:
                        yield DialogueChunk(piece, "Narrator", ((piece, "Narrator"),))
                    buffer = buffer[max_chars:].strip()
        if buffer.strip():
            yield DialogueChunk(buffer.strip(), "Narrator", ((buffer.strip(), "Narrator"),))

    # ── Multi-Speaker Chunking (Professional Engine) ────────────────

    @staticmethod
    def _chunk_multi_speaker(
        text: str, max_chars: int
    ) -> Generator[DialogueChunk, None, None]:
        """Chunk text with professional multi-pass speaker detection.

        The SpeakerDetectionEngine runs 10 passes for speaker assignment.
        Sentences are then grouped into DialogueChunks keeping
        multi-speaker dialogue together up to max_chars.
        """
        engine = SpeakerDetectionEngine()
        assignments = engine.detect_speakers(text)
        registry = engine.registry

        # Build sentence-level segments from assignments
        segments: List[Tuple[str, str]] = []
        sentences = SpeakerDetectionEngine._sentence_split(text)

        # Match assignments to sentences
        for i, sent in enumerate(sentences):
            if i < len(assignments):
                speaker = assignments[i].speaker
            else:
                # Fallback for any remaining sentences
                speaker = "Narrator"

            if sent.strip():
                segments.append((sent.strip(), speaker))

        # Group segments into chunks respecting max_chars
        buffer_segments: List[Tuple[str, str]] = []
        buffer_len = 0

        for text_seg, speaker in segments:
            if buffer_segments and (buffer_len + len(text_seg) > max_chars):
                # Yield current buffer
                yield TextProcessor._build_chunk(buffer_segments)
                buffer_segments = []
                buffer_len = 0

            buffer_segments.append((text_seg, speaker))
            buffer_len += len(text_seg)

        if buffer_segments:
            yield TextProcessor._build_chunk(buffer_segments)

    @staticmethod
    def _build_chunk(segments: List[Tuple[str, str]]) -> DialogueChunk:
        """Build a DialogueChunk from a list of (text, speaker) segments."""
        full_text = " ".join([s for s, _ in segments])
        speaker_counts: Dict[str, int] = {}
        for _, spk in segments:
            speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
        dominant = max(speaker_counts, key=speaker_counts.get)
        return DialogueChunk(full_text, dominant, tuple(segments))

    # ── Internals ───────────────────────────────────────────────────

    @staticmethod
    def _collect_text(source: str, is_raw_text: bool) -> str:
        stream = (
            TextProcessor._stream_raw_text(source)
            if is_raw_text
            else TextProcessor._stream_raw_file(source)
        )
        return "".join(stream)

    @staticmethod
    def _clean_stream(text_stream) -> Generator[str, None, None]:
        url_re = re.compile(r'https?://\S+|www\.\S+|[\w\.-]+@[\w\.-]+')
        for chunk in text_stream:
            chunk = url_re.sub('', chunk)
            chunk = re.sub(r'-\s*\n\s*', '', chunk)
            chunk = re.sub(r'\s+', ' ', chunk)
            if chunk.strip():
                yield chunk

    @staticmethod
    def _stream_raw_file(filepath: str) -> Generator[str, None, None]:
        ext = Path(filepath).suffix.lower()
        if ext == '.txt':
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                while block := f.read(65536): yield block
        elif ext == '.pdf':
            if not PDF_OK: raise ImportError("pip install pypdf")
            for page in PdfReader(filepath).pages:
                if text := page.extract_text(): yield text + "\n"
        elif ext == '.docx':
            if not DOCX_OK: raise ImportError("pip install python-docx")
            for para in Document(filepath).paragraphs:
                if para.text.strip(): yield para.text + "\n"
        elif ext in ('.md', '.html', '.htm'):
            if not BS4_OK: raise ImportError("pip install beautifulsoup4")
            content = Path(filepath).read_text(encoding='utf-8', errors='replace')
            if ext != '.md':
                soup = BeautifulSoup(content, 'html.parser')
                for tag in soup(["script", "style"]): tag.decompose()
                yield soup.get_text(separator="\n")
            else: yield content
        elif ext == '.epub':
            if not EPUB_OK: raise ImportError("pip install ebooklib")
            if not BS4_OK: raise ImportError("pip install beautifulsoup4")
            book = epub.read_epub(filepath)
            for item in book.get_items():
                if item.get_type() == 9:
                    try:
                        soup = BeautifulSoup(item.get_content(), 'html.parser')
                        for tag in soup(["script", "style"]): tag.decompose()
                        if text := soup.get_text(separator='\n').strip(): yield text + "\n"
                    except Exception: continue
        else: raise ValueError(f"Unsupported format: {ext}")

    @staticmethod
    def _stream_raw_text(raw: str) -> Generator[str, None, None]:
        bs = 8192
        for i in range(0, len(raw), bs):
            yield raw[i:i + bs]
