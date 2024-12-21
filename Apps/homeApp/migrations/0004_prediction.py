# Generated by Django 3.1.7 on 2024-11-29 14:01

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('homeApp', '0003_auto_20230817_1943'),
    ]

    operations = [
        migrations.CreateModel(
            name='Prediction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('url', models.URLField()),
                ('date', models.DateTimeField(default=django.utils.timezone.now)),
                ('prediction_result', models.CharField(max_length=50)),
                ('extracted_features', models.JSONField()),
                ('recommendation', models.TextField()),
            ],
        ),
    ]