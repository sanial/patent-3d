export interface ComponentData {
  name: string;
  color: string;
  claims: number[];
  desc: string;
  props: Record<string, string>;
}

export const PATENT_DATA: Record<string, ComponentData> = {
  '101': {
    name: 'Left Sensor Module',
    color: '#4f98a3',
    claims: [1, 3, 7],
    desc: 'Angled optical sensor unit positioned on the left side of the scanning array. Emits and receives structured light beams toward the plant tray reference line.',
    props: {
      'Wavelength': '850 nm IR',
      'Field of View': '42°',
      'Tilt Angle': '35°',
      'Interface': 'I²C / SPI',
    },
  },
  '102': {
    name: 'Right Sensor Module',
    color: '#a34f98',
    claims: [1, 3, 8],
    desc: 'Mirror-image optical sensor unit on the right side. Together with module 101, provides stereo depth measurements across the full tray width.',
    props: {
      'Wavelength': '850 nm IR',
      'Field of View': '42°',
      'Tilt Angle': '-35°',
      'Interface': 'I²C / SPI',
    },
  },
  '105': {
    name: 'Plant Tray',
    color: '#c8b87a',
    claims: [2, 5],
    desc: 'Standardised cultivation tray that acts as the measurement substrate. Raised side rails serve as reference datums for height normalisation.',
    props: {
      'Material': 'UV-stable ABS',
      'Dimensions': '720 × 100 mm',
      'Cell Count': '7',
      'Load Capacity': '4 kg',
    },
  },
  '108': {
    name: 'Reference Baseline',
    color: '#7ab87a',
    claims: [4, 6],
    desc: 'Virtual reference line defined by the sensor pair at tray surface level. All plant-height measurements are taken relative to this baseline.',
    props: {
      'Position': 'Y = 0 (tray surface)',
      'Span': '720 mm',
      'Update Rate': '30 Hz',
      'Accuracy': '±0.5 mm',
    },
  },
  '130': {
    name: 'Processing Controller',
    color: '#f0a050',
    claims: [1, 2, 9, 10],
    desc: 'Central embedded controller that aggregates sensor data, runs the growth-rate algorithm, and exposes results via a wireless network interface.',
    props: {
      'CPU': 'ARM Cortex-A53',
      'RAM': '512 MB',
      'Connectivity': 'Wi-Fi 802.11ac',
      'Power': '5 V / 2.4 A',
    },
  },
};
