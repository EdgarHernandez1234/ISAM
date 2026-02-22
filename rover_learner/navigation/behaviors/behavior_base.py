"""
rover_learner.navigation.behaviors.behavior_base

A thin interface for pluggable navigation behaviors.

Behaviors:
- consume NavObservation
- output NavProposal (Twist2D + status/reasons)
- maintain their own internal state, but remain deterministic and unit-testable
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..types import NavObservation, NavProposal


class Behavior(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (called on mode changes)."""
        raise NotImplementedError

    @abstractmethod
    def step(self, obs: NavObservation) -> NavProposal:
        """
        Compute one-step navigation proposal given the current observation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
