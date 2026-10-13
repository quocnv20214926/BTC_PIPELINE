"""Đã thay thế. Worker `model-signals` (model v2 5m) và `model-signals-v3` không còn được dùng.

Chiến lược live hiện tại là chiến lược H1 (touch + hướng B): xem pipeline/strategy_live.py
(`python -m pipeline strategy-h1`) và bộ thực thi pipeline/strategy_exec.py. File này giữ lại chỉ để các
import cũ báo lỗi rõ ràng; có thể xoá.
"""

raise ImportError('pipeline.model_signals đã bị thay bằng pipeline.strategy_live (python -m pipeline strategy-h1)')
