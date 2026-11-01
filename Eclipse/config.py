# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2020-2026 The ECLIPSE authors (CRAAM, Universidade Presbiteriana Mackenzie)
import sys
import os

# Adiciona a raiz do projeto ao path automaticamente
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)