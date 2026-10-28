"""Phase 5 — Knowledge base: the petrologist's textbook table (§11).

Module: src/rotapetro/knowledge.py
Files:  knowledge/minerals.csv, aliases.csv, hierarchy.yaml, likelihoods.yaml

Provides :class:`KnowledgeBase`, the machine-readable knowledge used by the
rule engine (Phase 6), the explanation generator (Phase 11), and at inference.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _txt(value, default: str) -> str:
    """Lower-case text of a CSV cell; pandas turns "n/a" into NaN."""
    if value is None or (isinstance(value, float) and value != value):
        return default
    return str(value).strip().lower()


class KnowledgeBase:
    """Machine-readable optical-mineralogy knowledge (§11).

    Parameters
    ----------
    knowledge_dir : Path
        Directory containing ``minerals.csv``, ``aliases.csv``,
        ``hierarchy.yaml``, and ``likelihoods.yaml``.

    Raises
    ------
    FileNotFoundError
        If ``minerals.csv`` or ``hierarchy.yaml`` is missing (these are
        mandatory).
    ValueError
        If validation fails (minerals not in hierarchy, dangling aliases).
    """

    def __init__(self, knowledge_dir: str | Path, cfg: dict | None = None) -> None:
        self.dir = Path(knowledge_dir)
        # Birefringence class boundaries must match Phase 2 (maps.biref_class_nm)
        self.biref_bounds = list((cfg or {}).get("maps", {}).get(
            "biref_class_nm", [100, 350, 1100, 2000]))
        self.unknown_names: set[str] = set()

        # ── Load files ──────────────────────────────────────────────
        minerals_path = self.dir / "minerals.csv"
        aliases_path = self.dir / "aliases.csv"
        hierarchy_path = self.dir / "hierarchy.yaml"
        likelihoods_path = self.dir / "likelihoods.yaml"

        if not minerals_path.is_file():
            raise FileNotFoundError(
                f"minerals.csv not found at {minerals_path}")
        if not hierarchy_path.is_file():
            raise FileNotFoundError(
                f"hierarchy.yaml not found at {hierarchy_path}")

        self.minerals_df = pd.read_csv(minerals_path)
        # Normalise column 'mineral' to lowercase + strip
        self.minerals_df["mineral"] = (
            self.minerals_df["mineral"].str.strip().str.lower())

        if aliases_path.is_file():
            self.aliases_df = pd.read_csv(aliases_path)
            self.aliases_df["ou_name"] = (
                self.aliases_df["ou_name"].str.strip().str.lower())
            self.aliases_df["canonical"] = (
                self.aliases_df["canonical"].str.strip().str.lower())
        else:
            logger.warning("aliases.csv not found — no aliases available")
            self.aliases_df = pd.DataFrame(
                columns=["ou_name", "canonical", "level"])

        with open(hierarchy_path, "r", encoding="utf-8") as fh:
            self.hierarchy = yaml.safe_load(fh) or {}

        if likelihoods_path.is_file():
            with open(likelihoods_path, "r", encoding="utf-8") as fh:
                self.likelihoods = yaml.safe_load(fh) or {}
        else:
            logger.warning("likelihoods.yaml not found — "
                           "rule engine will use uniform priors")
            self.likelihoods = {}

        # ── Build look-up structures ────────────────────────────────
        self._build_lookups()
        self._validate()
        logger.info("KnowledgeBase loaded: %d minerals, %d aliases, "
                     "%d groups",
                     len(self._mineral_dict),
                     len(self._alias_dict),
                     len(self._group_members))

    # ──────────────────────────────────────────────────────────────────
    #  Internal: build dictionaries
    # ──────────────────────────────────────────────────────────────────

    def _build_lookups(self) -> None:
        """Build internal dicts for fast access."""

        # mineral name → row dict
        self._mineral_dict: dict[str, dict] = {}
        for _, row in self.minerals_df.iterrows():
            name = row["mineral"]
            self._mineral_dict[name] = row.to_dict()

        # alias ou_name → (canonical, level)
        self._alias_dict: dict[str, tuple[str, str]] = {}
        for _, row in self.aliases_df.iterrows():
            self._alias_dict[row["ou_name"]] = (
                row["canonical"], row["level"])

        # Recursively parse hierarchy → mineral→group and group→members
        self._mineral_to_group: dict[str, str] = {}
        self._group_members: dict[str, list[str]] = {}
        # Separate occurrence list for duplicate detection (the dict
        # silently overwrites, masking duplicates)
        self._hierarchy_occurrences: list[str] = []
        # Also track group→immediate-parent for the super-group structure
        self._group_to_supergroup: dict[str, str] = {}

        self._parse_hierarchy(self.hierarchy, parent=None)

    def _parse_hierarchy(
        self, node: dict | list, parent: str | None,
    ) -> list[str]:
        """Walk the hierarchy tree recursively.

        The hierarchy can be nested dicts (super-groups containing
        sub-groups) or lists (leaf minerals).

        Example::

            felsic:
              silica: [quartz]
              feldspar:
                plagioclase: [plagioclase]
                k_feldspar: [orthoclase, microcline, sanidine]

        Here ``silica`` is a group with parent ``felsic``, and
        ``[quartz]`` are its member minerals.  ``feldspar`` is also
        registered as a group whose members are the union of
        ``plagioclase`` and ``k_feldspar`` members.

        Returns the list of minerals collected under this node (used
        to register intermediate groups).
        """
        collected: list[str] = []
        if isinstance(node, dict):
            for key, value in node.items():
                group_name = str(key)
                if parent is not None:
                    self._group_to_supergroup[group_name] = parent
                if isinstance(value, list):
                    # Leaf list: minerals belonging to this group
                    self._group_members.setdefault(group_name, [])
                    for mineral in value:
                        m = str(mineral).strip().lower()
                        self._hierarchy_occurrences.append(m)
                        self._mineral_to_group[m] = group_name
                        self._group_members[group_name].append(m)
                        collected.append(m)
                elif isinstance(value, dict):
                    # Nested sub-groups — recurse and collect
                    sub_minerals = self._parse_hierarchy(
                        value, parent=group_name)
                    # Register this intermediate group too
                    self._group_members.setdefault(group_name, [])
                    self._group_members[group_name].extend(sub_minerals)
                    collected.extend(sub_minerals)
                else:
                    logger.warning(
                        "Unexpected hierarchy value for %s: %s",
                        group_name, type(value))
        return collected

    # ──────────────────────────────────────────────────────────────────
    #  Internal: validation
    # ──────────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        """Validate knowledge base consistency (§11.5 rules)."""
        errors: list[str] = []

        # ── CSV schema validation (Fix B) ───────────────────────────
        required_cols = {
            "mineral", "group", "optic_class", "ppl_colour",
            "pleochroism", "relief", "biref_min", "biref_max",
            "colour_masks_biref", "extinction", "twinning",
            "cleavage_dirs", "habit", "absorption_vs_long",
            "twinkling", "source",
        }
        csv_cols = set(self.minerals_df.columns)
        missing_cols = required_cols - csv_cols
        if missing_cols:
            errors.append(f"minerals.csv missing columns: "
                          f"{sorted(missing_cols)}")

        # Duplicate mineral names in CSV
        csv_minerals = list(self.minerals_df["mineral"])
        csv_mineral_set = set(csv_minerals)
        if len(csv_minerals) != len(csv_mineral_set):
            from collections import Counter
            dup_csv = [m for m, c in Counter(csv_minerals).items() if c > 1]
            errors.append(f"Duplicate rows in minerals.csv: {dup_csv}")

        # Allowed enum values
        allowed_optic = {"uniaxial", "biaxial", "isotropic", "opaque"}
        for _, row in self.minerals_df.iterrows():
            oc = str(row.get("optic_class", "")).strip().lower()
            if oc not in allowed_optic:
                errors.append(
                    f"{row['mineral']}: invalid optic_class '{oc}' "
                    f"(allowed: {allowed_optic})")

        # Numeric biref ranges
        for _, row in self.minerals_df.iterrows():
            name = row["mineral"]
            if row.get("optic_class") == "opaque":
                continue
            try:
                bmin = float(row.get("biref_min", 0))
                bmax = float(row.get("biref_max", 0))
                if bmin < 0 or bmax < 0:
                    errors.append(f"{name}: negative birefringence")
                if bmin > bmax:
                    errors.append(
                        f"{name}: biref_min ({bmin}) > biref_max ({bmax})")
            except (TypeError, ValueError):
                # opaque may have empty biref — that's OK
                if str(row.get("optic_class", "")).lower() != "opaque":
                    errors.append(f"{name}: non-numeric birefringence")

        # ── Hierarchy validation (Fix A) ────────────────────────────
        hierarchy_minerals = set(self._mineral_to_group.keys())

        missing_from_hierarchy = csv_mineral_set - hierarchy_minerals
        if missing_from_hierarchy:
            errors.append(
                f"Minerals in CSV but not in hierarchy: "
                f"{sorted(missing_from_hierarchy)}")

        extra_in_hierarchy = hierarchy_minerals - csv_mineral_set
        if extra_in_hierarchy:
            errors.append(
                f"Minerals in hierarchy but not in CSV: "
                f"{sorted(extra_in_hierarchy)}")

        # Check uniqueness: use _hierarchy_occurrences (list), NOT
        # the dict (which silently overwrites duplicates)
        from collections import Counter
        occ_counts = Counter(self._hierarchy_occurrences)
        dupes = {m for m, c in occ_counts.items() if c > 1}
        if dupes:
            errors.append(f"Minerals appearing more than once in "
                          f"hierarchy: {sorted(dupes)}")

        # ── Alias validation ────────────────────────────────────────
        all_known = csv_mineral_set | set(self._group_members.keys())
        for _, row in self.aliases_df.iterrows():
            target = row["canonical"]
            if target not in all_known:
                errors.append(
                    f"Alias '{row['ou_name']}' -> '{target}' "
                    f"but '{target}' is neither a mineral nor a group")

        if errors:
            for e in errors:
                logger.error("Knowledge validation: %s", e)
            raise ValueError(
                f"Knowledge base validation failed with {len(errors)} "
                f"error(s). See log for details.")

    # ──────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────

    def resolve_alias(self, name: str) -> tuple[str, str]:
        """Resolve an OU mineral name to (canonical, level).

        Parameters
        ----------
        name : str
            The name as it appears in OU specimen metadata.

        Returns
        -------
        (canonical, level)
            ``canonical`` is the mineral or group name used internally.
            ``level`` is ``'mineral'`` or ``'group'``.
            If *name* is not an alias, returns ``(name, 'mineral')``.
        """
        key = name.strip().lower()
        if key in self._alias_dict:
            return self._alias_dict[key]
        # Not an alias — assume it's a direct mineral name
        return key, "mineral"

    def is_true_alias(self, name: str) -> bool:
        """V1: True if *name* is a true spelling alias (e.g. albite→plagioclase),
        False if it is a family/group membership mapping (e.g. feldspar→feldspar group).

        Used to separate alias-level resolution from family membership in
        candidate-set construction and diagnostic reports.
        """
        key = name.strip().lower()
        if key not in self._alias_dict:
            return False
        canonical, level = self._alias_dict[key]
        # True alias: maps to a specific mineral at mineral level
        # Family membership: maps to a group at group level
        return level == "mineral" and canonical != key

    def expand_group(self, group: str) -> list[str]:
        """Expand a group name to all its member minerals (§11.5).

        Recursively collects minerals from sub-groups.
        If *group* is a mineral (not a group), returns ``[group]``.
        """
        group = group.strip().lower()

        # Direct group lookup
        if group in self._group_members:
            result: list[str] = list(self._group_members[group])
            return result

        # Maybe it's a super-group containing sub-groups
        if group in self.hierarchy:
            return self._collect_minerals(self.hierarchy[group])

        # Check if it's a nested super-group (e.g. 'felsic')
        for key, val in self.hierarchy.items():
            if key == group:
                return self._collect_minerals(val)

        # Not a group — might be a mineral
        if group in self._mineral_dict:
            return [group]

        logger.warning("expand_group: unknown group '%s'", group)
        return []

    def _collect_minerals(self, node: dict | list) -> list[str]:
        """Recursively collect all mineral names from a hierarchy node."""
        if isinstance(node, list):
            return [str(m).strip().lower() for m in node]
        if isinstance(node, dict):
            result: list[str] = []
            for val in node.values():
                result.extend(self._collect_minerals(val))
            return result
        return []

    def get_mineral(self, name: str) -> dict:
        """Get the full mineral property dict from minerals.csv.

        Returns an empty dict if *name* is not found.
        """
        return dict(self._mineral_dict.get(name.strip().lower(), {}))

    def get_group(self, mineral: str) -> str:
        """Return the canonical group name for a mineral.

        Uses the ``group`` column from ``minerals.csv`` (the spec's
        intended group representation), falling back to the hierarchy
        leaf node if the mineral is not in the CSV.

        Returns ``'unknown'`` if the mineral is not found.
        """
        key = mineral.strip().lower()
        # Primary: minerals.csv 'group' column
        m_info = self._mineral_dict.get(key)
        if m_info and "group" in m_info:
            g = str(m_info["group"]).strip().lower()
            if g:
                return g
        # Fallback: hierarchy leaf
        return self._mineral_to_group.get(key, "unknown")

    def get_candidate_set(
        self,
        rock_forming: list[str],
        accessory: list[str] | None = None,
    ) -> set[str]:
        """Build the candidate mineral set for a specimen (§12.1).

        1. Resolve every OU name through aliases.
        2. Expand group names to member minerals.
        3. Always add ``opaque`` and ``fine_aggregate``.

        Returns a set of canonical mineral names (all lowercase).
        """
        if accessory is None:
            accessory = []

        raw_names = list(rock_forming) + list(accessory)
        candidates: set[str] = set()

        for name in raw_names:
            canonical, level = self.resolve_alias(name)
            # Expand if it's a known group (from alias level OR from
            # the hierarchy itself)
            if level == "group" or canonical in self._group_members:
                expanded = self.expand_group(canonical)
                if expanded:
                    candidates.update(expanded)
                    continue
            if canonical in self._mineral_dict:
                candidates.add(canonical)
                continue
            # Not in minerals.csv / hierarchy: record for diagnostics
            if canonical not in self.unknown_names:
                logger.warning("Unknown OU mineral name '%s' — add it to "
                               "aliases.csv (or minerals.csv)", name)
            self.unknown_names.add(canonical)

        # Always include opaque and fine_aggregate
        candidates.add("opaque")
        candidates.add("fine_aggregate")

        return candidates

    def all_minerals(self) -> list[str]:
        """Return a sorted list of all canonical mineral names."""
        return sorted(self._mineral_dict.keys())

    def all_groups(self) -> dict[str, list[str]]:
        """Return group→[minerals] mapping."""
        return dict(self._group_members)

    def get_observation_weights(self) -> dict[str, float]:
        """Return observation term weights from likelihoods.yaml."""
        return dict(self.likelihoods.get("observation_weights", {}))

    def get_likelihoods(self) -> dict:
        """Return P(obs|mineral) settings from likelihoods.yaml."""
        return dict(self.likelihoods.get("p_obs_given_mineral", {}))

    def get_biref_classes(self, mineral: str) -> list[int]:
        """Compute expected birefringence classes from δ range.

        Uses biref_min, biref_max and standard 30 µm thickness:
            retardation = δ × 30000 nm
        Classes: C0 (<100), C1 (100-350), C2 (350-1100),
                 C3 (1100-2000), C4 (>2000).
        """
        m = self.get_mineral(mineral)
        if not m:
            return [0]

        biref_min = m.get("biref_min", 0.0)
        biref_max = m.get("biref_max", 0.0)

        # Handle NaN / missing
        try:
            biref_min = float(biref_min)
            biref_max = float(biref_max)
        except (TypeError, ValueError):
            return [0]

        # Retardation range at 30 µm (but we use 25-35 µm range for
        # conservative estimate)
        gamma_min = biref_min * 25000  # thin edge
        gamma_max = biref_max * 35000  # thick edge

        boundaries = self.biref_bounds
        classes: set[int] = set()

        for gamma in [gamma_min, gamma_max]:
            cls = 0
            for i, b in enumerate(boundaries):
                if gamma >= b:
                    cls = i + 1
            classes.add(cls)

        # Also add intermediate classes
        result = list(range(min(classes), max(classes) + 1))
        return result if result else [0]

    def get_ancestors(self, mineral: str) -> list[str]:
        """Groups above a mineral, nearest first, skipping any group that
        has the same name as the mineral (e.g. plagioclase -> [feldspar,
        felsic]; microcline -> [k_feldspar, feldspar, felsic])."""
        m = mineral.strip().lower()
        out: list[str] = []
        g = self._mineral_to_group.get(m)
        while g is not None:
            if g != m:
                out.append(g)
            g = self._group_to_supergroup.get(g)
        return out

    # Phase-4 vocabulary <-> textbook vocabulary --------------------------
    _COLOUR_MAP = {            # textbook colour -> Phase-4 colour names
        "colourless": {"colourless", "grey"}, "cloudy": {"colourless", "grey"},
        "brown": {"brown", "yellow"}, "red-brown": {"brown", "pink/red"},
        "yellow": {"yellow", "brown"}, "green": {"green"},
        "blue-green": {"green", "blue"}, "blue": {"blue"},
        "pink": {"pink/red"}, "purple": {"pink/red", "blue"},
        "black": {"black"},
    }
    _HABIT_MAP = {             # textbook habit -> Phase-4 aspect classes
        "equant": {"equant"}, "rhombic": {"equant"},
        "anhedral": {"equant", "prismatic"}, "euhedral": {"equant", "prismatic"},
        "tabular": {"prismatic", "equant"}, "prismatic": {"prismatic", "acicular"},
        "platy": {"prismatic", "acicular"}, "fibrous": {"acicular", "prismatic"},
        "acicular": {"acicular"},
    }

    @staticmethod
    def normalize_obs(field: str, value) -> str:
        """Normalise a Phase-4 value to the vocabulary used here."""
        v = str(value).strip().lower()
        if field == "relief":
            return v.replace(" ", "_")          # "very high" -> "very_high"
        return v

    def expected_obs(self, mineral: str) -> dict:
        """Textbook expectations of a mineral expressed in the Phase-4
        feature vocabulary (what the rule engine compares against).

        Keys: optic_class, ppl_colour, pleochroism, relief, biref_cls,
        colour_masks_biref, extinction (parallel/inclined/symmetric/
        undulose/none), ext_angle_max_deg, twinning, cleavage_dirs,
        cleavage_acute_deg (list of (lo, hi)), habit (aspect classes),
        absorption_vs_long, twinkling.
        """
        m = self.get_mineral(mineral)
        if not m:
            return {}
        lst = self.parse_list_field
        oc = str(m.get("optic_class", "")).lower()
        colours: set[str] = set()
        for c in lst(m.get("ppl_colour")):
            colours |= self._COLOUR_MAP.get(c.lower(), {c.lower()})
        habits: set[str] = set()
        for h in lst(m.get("habit")):
            habits |= self._HABIT_MAP.get(h.lower(), set())
        acute: list[tuple[float, float]] = []
        for rng in lst(m.get("cleavage_angle_deg")):
            try:
                a, b = (float(x) for x in rng.split("-"))
                # V1 FIX: convert range to acute cleavage angles [0-90°].
                # Phase 4 reports angles in 0-90°.  The textbook range
                # "56-124" means the acute pair is 56° and (180-124)=56°,
                # but "87-93" means 87° and 87°.  We fold each endpoint
                # to 0-90 individually, then sort to get (lo, hi).
                a_acute = min(a, 180.0 - a)
                b_acute = min(b, 180.0 - b)
                lo, hi = sorted((a_acute, b_acute))
                # Expand range by ±5° tolerance for measurement noise
                acute.append((max(0.0, lo - 5.0), min(90.0, hi + 5.0)))
            except ValueError:
                pass
        try:
            ext_max = float(m.get("ext_angle_max_deg"))
        except (TypeError, ValueError):
            ext_max = float("nan")
        try:
            cdirs = int(float(m.get("cleavage_dirs", 0)))
        except (TypeError, ValueError):
            cdirs = 0
        return {
            "optic_class": ("anisotropic" if oc in ("uniaxial", "biaxial") else oc),
            "ppl_colour": colours,
            "pleochroism": {p.lower() for p in lst(m.get("pleochroism"))},
            "relief": {r.lower() for r in lst(m.get("relief"))},
            "biref_cls": set(self.get_biref_classes(mineral)),
            "colour_masks_biref": _txt(m.get("colour_masks_biref"), "no"),
            "extinction": {e.lower() for e in lst(m.get("extinction"))},
            "ext_angle_max_deg": ext_max,
            "twinning": {t.lower() for t in lst(m.get("twinning"))},
            "cleavage_dirs": min(cdirs, 2),
            "cleavage_acute_deg": acute,
            "habit": habits,
            "absorption_vs_long": _txt(m.get("absorption_vs_long"), "n/a"),
            "twinkling": _txt(m.get("twinkling"), "no") == "yes",
        }

    def parse_list_field(self, value: str | float) -> list[str]:
        """Parse a semicolon-separated CSV field to a list of strings.

        Handles NaN / None gracefully.
        """
        if pd.isna(value) or value is None:
            return []
        return [v.strip() for v in str(value).split(";") if v.strip()]

    def __repr__(self) -> str:
        return (f"KnowledgeBase(minerals={len(self._mineral_dict)}, "
                f"aliases={len(self._alias_dict)}, "
                f"groups={len(self._group_members)})")
