"""Create five auditable 20-candle PNG examples from the historical 1h CSV."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SOURCE = Path('historical_data/BTCUSDT_1h_all.csv')
OUTPUT = Path('artifacts/candlestick_examples_1h')
WIDTH, HEIGHT = 1600, 900
LEFT, RIGHT, TOP, PRICE_BOTTOM = 120, 70, 95, 680
VOLUME_TOP, VOLUME_BOTTOM = 715, 820


def font(size, bold=False):
    names = ['arialbd.ttf', 'Arial Bold.ttf'] if bold else ['arial.ttf', 'Arial.ttf']
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def timestamp(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M UTC')


def draw_chart(rows, target, number):
    image = Image.new('RGB', (WIDTH, HEIGHT), '#0d1117')
    draw = ImageDraw.Draw(image)
    title_font, label_font, small_font = font(31, True), font(20), font(17)
    draw.text((LEFT, 30), f'BTCUSDT · 1h · Sample {number:02d} · 20 candles',
              fill='#e6edf3', font=title_font)

    low = min(row['low'] for row in rows)
    high = max(row['high'] for row in rows)
    padding = max((high - low) * 0.08, high * 0.0005)
    pmin, pmax = low - padding, high + padding
    chart_width = WIDTH - LEFT - RIGHT
    candle_step = chart_width / len(rows)
    body_width = max(8, int(candle_step * 0.56))

    def y_price(value):
        return TOP + (pmax - value) / (pmax - pmin) * (PRICE_BOTTOM - TOP)

    for i in range(6):
        price = pmin + (pmax - pmin) * i / 5
        y = y_price(price)
        draw.line((LEFT, y, WIDTH - RIGHT, y), fill='#26303b', width=1)
        text = f'{price:,.0f}'
        box = draw.textbbox((0, 0), text, font=small_font)
        draw.text((LEFT - 15 - (box[2] - box[0]), y - 10), text,
                  fill='#8b949e', font=small_font)

    max_volume = max(row['volume'] for row in rows) or 1
    for i, row in enumerate(rows):
        x = LEFT + candle_step * (i + 0.5)
        rising = row['close'] >= row['open']
        color = '#26a69a' if rising else '#ef5350'
        draw.line((x, y_price(row['high']), x, y_price(row['low'])), fill=color, width=3)
        y_open, y_close = y_price(row['open']), y_price(row['close'])
        y1, y2 = sorted((y_open, y_close))
        if y2 - y1 < 2:
            y2 = y1 + 2
        draw.rectangle((x - body_width / 2, y1, x + body_width / 2, y2),
                       fill=color, outline=color)
        volume_height = row['volume'] / max_volume * (VOLUME_BOTTOM - VOLUME_TOP)
        draw.rectangle((x - body_width / 2, VOLUME_BOTTOM - volume_height,
                        x + body_width / 2, VOLUME_BOTTOM), fill=color)

    draw.line((LEFT, PRICE_BOTTOM, WIDTH - RIGHT, PRICE_BOTTOM), fill='#56606b', width=1)
    draw.line((LEFT, VOLUME_BOTTOM, WIDTH - RIGHT, VOLUME_BOTTOM), fill='#56606b', width=1)
    draw.text((25, VOLUME_TOP + 30), 'Volume', fill='#8b949e', font=small_font)

    tick_indexes = [0, 4, 9, 14, 19]
    for idx in tick_indexes:
        x = LEFT + candle_step * (idx + 0.5)
        label = datetime.fromtimestamp(rows[idx]['open_time_ms']/1000, timezone.utc).strftime('%m-%d\n%H:%M')
        draw.multiline_text((x - 28, VOLUME_BOTTOM + 12), label, fill='#8b949e',
                            font=small_font, spacing=2, align='center')
    subtitle = f"{timestamp(rows[0]['open_time_ms'])}  →  {timestamp(rows[-1]['open_time_ms'])}"
    draw.text((LEFT, 66), subtitle, fill='#8b949e', font=label_font)
    image.save(target, 'PNG', optimize=True)


def main():
    with SOURCE.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        rows = [dict(open_time_ms=int(r['open_time_ms']), open=float(r['open']),
                     high=float(r['high']), low=float(r['low']), close=float(r['close']),
                     volume=float(r['volume'])) for r in reader]
    if len(rows) < 100:
        raise ValueError('Need at least 100 candles')
    selected = rows[-100:]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {'source': str(SOURCE), 'timeframe': '1h', 'selection': 'last 100 rows',
                'candles_per_image': 20, 'images': []}
    for index in range(5):
        sample = selected[index * 20:(index + 1) * 20]
        name = f'btc_1h_sample_{index + 1:02d}.png'
        draw_chart(sample, OUTPUT / name, index + 1)
        manifest['images'].append({'file': name, 'rows': len(sample),
            'first_open_time_ms': sample[0]['open_time_ms'],
            'first_open_time_utc': timestamp(sample[0]['open_time_ms']),
            'last_open_time_ms': sample[-1]['open_time_ms'],
            'last_open_time_utc': timestamp(sample[-1]['open_time_ms']),
            'min_low': min(r['low'] for r in sample),
            'max_high': max(r['high'] for r in sample)})
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
