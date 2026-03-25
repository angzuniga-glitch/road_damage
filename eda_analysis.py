import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

DATA_DIR = "/nfshome/data/"

def parse_xml_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = {
        'filename': root.find('filename').text if root.find('filename') is not None else None,
        'width': int(root.find('size/width').text) if root.find('size/width') is not None else None,
        'height': int(root.find('size/height').text) if root.find('size/height') is not None else None,
        'depth': int(root.find('size/depth').text) if root.find('size/depth') is not None else None,
        'objects': []
    }

    for obj in root.findall('object'):
        name = obj.find('name')
        bndbox = obj.find('bndbox')
        if name is not None and bndbox is not None:
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            data['objects'].append({
                'class': name.text,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'width': xmax - xmin,
                'height': ymax - ymin,
                'area': (xmax - xmin) * (ymax - ymin)
            })

    return data

def analyze_dataset():
    countries = ['China_Drone', 'China_MotorBike', 'Czech', 'India', 'Japan', 'Norway', 'United_States']

    results = {
        'dataset_overview': [],
        'class_distribution': defaultdict(lambda: defaultdict(int)),
        'bbox_stats': defaultdict(list),
        'image_stats': defaultdict(list),
        'objects_per_image': defaultdict(list)
    }
    for country in countries:
        country_path = os.path.join(DATA_DIR, country)
        print(f"\nStarting {country} EDA")

        xml_files = list(Path(country_path).rglob("*.xml"))
        image_files = list(Path(country_path).rglob("*.jpg")) + list(Path(country_path).rglob("*.png"))

        train_images = len([f for f in image_files if 'train' in str(f)])
        test_images = len([f for f in image_files if 'test' in str(f)])

        results['dataset_overview'].append({
            'Country': country,
            'Total Images': len(image_files),
            'Train Images': train_images,
            'Test Images': test_images,
            'Total Annotations': len(xml_files)
        })

        total_objects = 0
        for xml_file in xml_files[:min(len(xml_files), 1000)]:
            try:
                data = parse_xml_annotation(xml_file)

                num_objects = len(data['objects'])
                total_objects += num_objects
                results['objects_per_image'][country].append(num_objects)

                if data['width'] and data['height']:
                    results['image_stats'][country].append({
                        'width': data['width'],
                        'height': data['height'],
                        'aspect_ratio': data['width'] / data['height']
                    })

                for obj in data['objects']:
                    results['class_distribution'][country][obj['class']] += 1
                    results['bbox_stats'][country].append({
                        'width': obj['width'],
                        'height': obj['height'],
                        'area': obj['area'],
                        'aspect_ratio': obj['width'] / obj['height'] if obj['height'] > 0 else 0
                    })
            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
                continue

        print(f"Images: {len(image_files)}, Annotations: {len(xml_files)}, Sample Objects: {total_objects}")

    return results

def generate_visualizations(results):
    df_overview = pd.DataFrame(results['dataset_overview'])
    print(df_overview.to_string(index=False))
    print(f"\nTotal Images: {df_overview['Total Images'].sum()}")
    print(f"Total Annotations: {df_overview['Total Annotations'].sum()}")

    all_classes = set()
    for country_classes in results['class_distribution'].values():
        all_classes.update(country_classes.keys())

    class_data = []
    for country, classes in results['class_distribution'].items():
        for cls, count in classes.items():
            class_data.append({'Country': country, 'Class': cls, 'Count': count})

    df_classes = pd.DataFrame(class_data)

    if not df_classes.empty:
        print("\nError: Damage Classes Found")
        class_summary = df_classes.groupby('Class')['Count'].sum().sort_values(ascending=False)
        for cls, count in class_summary.items():
            print(f"  {cls}: {count:,}")

        print("\nClass Distribution by Country")
        class_pivot = df_classes.pivot_table(index='Country', columns='Class', values='Count', fill_value=0)
        print(class_pivot)

        plt.figure(figsize=(16, 10))

        plt.subplot(2, 2, 1)
        class_summary.plot(kind='bar')
        plt.title('Overall Damage Class Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Damage Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.subplot(2, 2, 2)
        if len(df_classes) > 0:
            df_plot = df_classes.groupby(['Country', 'Class'])['Count'].sum().unstack(fill_value=0)
            df_plot.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.title('Class Distribution by Country', fontsize=14, fontweight='bold')
            plt.xlabel('Country')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.subplot(2, 2, 3)
        obj_per_img_data = []
        for country, counts in results['objects_per_image'].items():
            if counts:
                obj_per_img_data.extend([{'Country': country, 'Objects': c} for c in counts])

        if obj_per_img_data:
            df_obj = pd.DataFrame(obj_per_img_data)
            df_obj.boxplot(by='Country', column='Objects', ax=plt.gca())
            plt.title('Objects per Image Distribution by Country', fontsize=14, fontweight='bold')
            plt.suptitle('')
            plt.xlabel('Country')
            plt.ylabel('Number of Objects')
            plt.xticks(rotation=45)

        plt.subplot(2, 2, 4)
        bbox_areas = []
        for country, bboxes in results['bbox_stats'].items():
            if bboxes:
                areas = [b['area'] for b in bboxes if b['area'] > 0]
                if areas:
                    bbox_areas.extend([{'Country': country, 'Area': a} for a in areas[:500]])

        if bbox_areas:
            df_bbox = pd.DataFrame(bbox_areas)
            df_bbox['Log Area'] = np.log10(df_bbox['Area'] + 1)
            df_bbox.boxplot(by='Country', column='Log Area', ax=plt.gca())
            plt.title('Bounding Box Area Distribution Log Scale', fontsize=14, fontweight='bold')
            plt.suptitle('')
            plt.xlabel('Country')
            plt.ylabel('Log10(Area)')
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
        print("\nVisual saved to eda_visualization.png")


    for country, img_stats in results['image_stats'].items():
        if img_stats:
            widths = [s['width'] for s in img_stats]
            heights = [s['height'] for s in img_stats]
            aspects = [s['aspect_ratio'] for s in img_stats]

            print(f"\n{country}:")
            print(f"  Image dimensions: {np.mean(widths):.0f}x{np.mean(heights):.0f} (avg)")
            print(f"  Width range: {min(widths)} - {max(widths)}")
            print(f"  Height range: {min(heights)} - {max(heights)}")
            print(f"  Aspect ratio: {np.mean(aspects):.2f} (avg)")

    for country, bboxes in results['bbox_stats'].items():
        if bboxes:
            widths = [b['width'] for b in bboxes]
            heights = [b['height'] for b in bboxes]
            areas = [b['area'] for b in bboxes]

            print(f"\n{country}:")
            print(f"  Avg bbox size: {np.mean(widths):.1f}x{np.mean(heights):.1f}")
            print(f"  Avg bbox area: {np.mean(areas):.0f} pixels")
            print(f"  Bbox area range: {min(areas)} - {max(areas)}")

    for country, counts in results['objects_per_image'].items():
        if counts:
            print(f"\n{country}:")
            print(f"  Avg objects per image: {np.mean(counts):.2f}")
            print(f"  Median: {np.median(counts):.0f}")
            print(f"  Range: {min(counts)} - {max(counts)}")
            print(f"  Images with no objects: {counts.count(0)}")

def main():
    print(f"Data directory: {DATA_DIR}")

    results = analyze_dataset()

    generate_visualizations(results)

if __name__ == "__main__":
    main()
