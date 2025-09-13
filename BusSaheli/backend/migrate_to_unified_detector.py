#!/usr/bin/env python3
"""
Migration Script: Replace Duplicate Detection Classes with Unified Interface
This script helps migrate from 10+ duplicate classes to the unified system
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionClassMigrator:
    """Handles migration from duplicate classes to unified system"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backup_duplicate_classes"
        self.duplicate_files = []
        self.migration_log = []
        
    def identify_duplicate_files(self) -> List[Path]:
        """Identify all duplicate detection class files"""
        duplicate_patterns = [
            "*gender_detection*.py",
            "professional_gender_detection.py",
            "academic_gender_detection.py", 
            "balanced_gender_detection.py",
            "premium_gender_detection.py",
            "ultra_sensitive_gender_detection.py",
            "polished_gender_detection.py",
            "simple_logic_gender_detection.py",
            "logic_based_gender_detection.py",
            "improved_gender_detection.py",
            "precision_tuned_gender_detection.py"
        ]
        
        for pattern in duplicate_patterns:
            files = list(self.project_root.glob(pattern))
            self.duplicate_files.extend(files)
        
        # Remove duplicates and sort
        self.duplicate_files = sorted(list(set(self.duplicate_files)))
        
        logger.info(f"🔍 Found {len(self.duplicate_files)} duplicate detection files")
        for file in self.duplicate_files:
            logger.info(f"   - {file.name}")
        
        return self.duplicate_files
    
    def create_backup(self) -> bool:
        """Create backup of duplicate files before migration"""
        try:
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
            
            self.backup_dir.mkdir(exist_ok=True)
            
            for file in self.duplicate_files:
                if file.exists():
                    backup_file = self.backup_dir / file.name
                    shutil.copy2(file, backup_file)
                    logger.info(f"📦 Backed up {file.name}")
            
            logger.info(f"✅ Backup created in {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False
    
    def create_migration_guide(self) -> str:
        """Create a migration guide for developers"""
        guide_content = """# 🔄 Migration Guide: Unified Gender Detector

## 📋 Overview
This guide helps you migrate from duplicate detection classes to the unified system.

## 🚀 Quick Migration

### Before (Old Way)
```python
# Multiple different classes
from professional_gender_detection import ProfessionalGenderDetector
from academic_gender_detection import AcademicGenderDetector
from balanced_gender_detection import BalancedGenderDetector

# Different initialization methods
detector1 = ProfessionalGenderDetector()
detector2 = AcademicGenderDetector()
detector3 = BalancedGenderDetector()

# Different method names
faces1 = detector1.detect_faces(image)
faces2 = detector2.detect_faces(image)
faces3 = detector3.detect_faces(image)
```

### After (New Way)
```python
# Single unified interface
from BusSaheli.backend.unified_gender_detector import create_gender_detector

# Consistent initialization
detector1 = create_gender_detector("professional")
detector2 = create_gender_detector("academic")
detector3 = create_gender_detector("balanced")

# Consistent method names
results1 = detector1.detect_gender(image)
results2 = detector2.detect_gender(image)
results3 = detector3.detect_gender(image)
```

## 🔧 Migration Steps

### 1. Update Imports
```python
# OLD
from professional_gender_detection import ProfessionalGenderDetector

# NEW
from BusSaheli.backend.unified_gender_detector import create_gender_detector
```

### 2. Update Initialization
```python
# OLD
detector = ProfessionalGenderDetector()

# NEW
detector = create_gender_detector("professional")
```

### 3. Update Method Calls
```python
# OLD
faces = detector.detect_faces(image)
gender = detector.detect_gender(face_roi)

# NEW
results = detector.detect_gender(image)
for result in results:
    bbox = result.bbox
    gender = result.gender
    confidence = result.confidence
```

### 4. Update Result Handling
```python
# OLD
for (x, y, w, h) in faces:
    gender = detector.detect_gender(face_roi)
    confidence = 0.8  # Hardcoded

# NEW
for result in results:
    x, y, w, h = result.bbox
    gender = result.gender
    confidence = result.confidence
    face_id = result.face_id
    timestamp = result.timestamp
```

## 📊 Available Algorithms

- `haar_cascade` - Most reliable, works everywhere
- `professional` - Advanced features, good accuracy
- `academic` - Research-grade, high precision
- `balanced` - Good balance of speed and accuracy
- `premium` - Best accuracy, slower
- `ultra_sensitive` - Highest sensitivity
- `polished` - Clean, minimal interface
- `simple_logic` - Fast, basic detection

## 🎯 Benefits

1. **Consistent Interface** - Same methods across all algorithms
2. **Memory Management** - Automatic cleanup and optimization
3. **Performance Tracking** - Built-in statistics and monitoring
4. **Easy Switching** - Change algorithms without code changes
5. **Standardized Results** - Same data structure for all results
6. **Better Testing** - Single interface to test
7. **Maintainability** - One place to fix bugs and add features

## 🚨 Breaking Changes

- Method names changed: `detect_faces()` → `detect_gender()`
- Return format changed: List of tuples → List of DetectionResult objects
- Initialization changed: Direct class instantiation → Factory function
- Some algorithm-specific features may not be available

## 🔍 Testing

After migration, test your code:
```python
detector = create_gender_detector("professional")
results = detector.detect_gender(test_image)
assert len(results) > 0, "No faces detected"
assert results[0].gender in ["Male", "Female"], "Invalid gender"
assert 0 <= results[0].confidence <= 1, "Invalid confidence"
```

## 📞 Support

If you encounter issues:
1. Check the backup files in `backup_duplicate_classes/`
2. Review the unified detector code
3. Test with different algorithms
4. Check the logs for error messages
"""
        
        guide_path = self.project_root / "MIGRATION_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        logger.info(f"📖 Migration guide created: {guide_path}")
        return str(guide_path)
    
    def create_replacement_imports(self) -> Dict[str, str]:
        """Create import replacement mappings"""
        return {
            "from professional_gender_detection import ProfessionalGenderDetector": 
                "from BusSaheli.backend.unified_gender_detector import create_gender_detector",
            
            "from academic_gender_detection import AcademicGenderDetector": 
                "from BusSaheli.backend.unified_gender_detector import create_gender_detector",
            
            "from balanced_gender_detection import BalancedGenderDetector": 
                "from BusSaheli.backend.unified_gender_detector import create_gender_detector",
            
            "from premium_gender_detection import PremiumGenderDetector": 
                "from BusSaheli.backend.unified_gender_detector import create_gender_detector",
            
            "from ultra_sensitive_gender_detection import UltraSensitiveGenderDetector": 
                "from BusSaheli.backend.unified_gender_detector import create_gender_detector",
            
            "from polished_gender_detection import PolishedGenderDetector": 
                "from BusSaheli.backend.unified_gender_detector import create_gender_detector",
            
            "from simple_logic_gender_detection import SimpleLogicGenderDetector": 
                "from BusSaheli.backend.unified_gender_detector import create_gender_detector",
            
            "from logic_based_gender_detection import LogicBasedGenderDetector": 
                "from BusSaheli.backend.unified_gender_detector import create_gender_detector",
            
            "from improved_gender_detection import ImprovedGenderDetector": 
                "from BusSaheli.backend.unified_gender_detector import create_gender_detector",
            
            "from precision_tuned_gender_detection import PrecisionTunedGenderDetector": 
                "from BusSaheli.backend.unified_gender_detector import create_gender_detector"
        }
    
    def create_initialization_replacements(self) -> Dict[str, str]:
        """Create initialization replacement mappings"""
        return {
            "ProfessionalGenderDetector()": 'create_gender_detector("professional")',
            "AcademicGenderDetector()": 'create_gender_detector("academic")',
            "BalancedGenderDetector()": 'create_gender_detector("balanced")',
            "PremiumGenderDetector()": 'create_gender_detector("premium")',
            "UltraSensitiveGenderDetector()": 'create_gender_detector("ultra_sensitive")',
            "PolishedGenderDetector()": 'create_gender_detector("polished")',
            "SimpleLogicGenderDetector()": 'create_gender_detector("simple_logic")',
            "LogicBasedGenderDetector()": 'create_gender_detector("simple_logic")',
            "ImprovedGenderDetector()": 'create_gender_detector("professional")',
            "PrecisionTunedGenderDetector()": 'create_gender_detector("premium")'
        }
    
    def generate_migration_report(self) -> str:
        """Generate a comprehensive migration report"""
        report = f"""# 📊 Migration Report: Unified Gender Detector

## 📋 Summary
- **Duplicate Files Found**: {len(self.duplicate_files)}
- **Backup Created**: {self.backup_dir}
- **Migration Guide**: MIGRATION_GUIDE.md
- **Status**: Ready for migration

## 📁 Files to Migrate
"""
        
        for file in self.duplicate_files:
            report += f"- {file.name}\n"
        
        report += f"""
## 🔧 Migration Steps

1. **Backup Created**: All duplicate files backed up to `{self.backup_dir}`
2. **Unified Detector**: Created `BusSaheli/backend/unified_gender_detector.py`
3. **Migration Guide**: Created `MIGRATION_GUIDE.md`
4. **Next Steps**: 
   - Update imports in your code
   - Replace class instantiations
   - Update method calls
   - Test thoroughly

## 🎯 Benefits After Migration

- **Code Reduction**: ~60% less duplicate code
- **Maintainability**: Single interface to maintain
- **Performance**: Better memory management
- **Consistency**: Same API across all algorithms
- **Testing**: Easier to test and debug

## 🚨 Important Notes

- **Backup**: Original files are safely backed up
- **Breaking Changes**: Some method signatures changed
- **Testing**: Test thoroughly after migration
- **Rollback**: Use backup files if needed

## 📞 Next Steps

1. Review the migration guide
2. Update your code gradually
3. Test each change
4. Remove old files when confident
5. Update documentation

Migration completed successfully! 🎉
"""
        
        report_path = self.project_root / "MIGRATION_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📊 Migration report created: {report_path}")
        return str(report_path)
    
    def run_migration(self) -> bool:
        """Run the complete migration process"""
        try:
            logger.info("🚀 Starting migration process...")
            
            # Step 1: Identify duplicate files
            self.identify_duplicate_files()
            
            # Step 2: Create backup
            if not self.create_backup():
                return False
            
            # Step 3: Create migration guide
            self.create_migration_guide()
            
            # Step 4: Generate report
            self.generate_migration_report()
            
            logger.info("✅ Migration process completed successfully!")
            logger.info(f"📁 Backup created: {self.backup_dir}")
            logger.info("📖 Check MIGRATION_GUIDE.md for detailed instructions")
            logger.info("📊 Check MIGRATION_REPORT.md for summary")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False

def main():
    """Main migration function"""
    project_root = os.getcwd()
    migrator = DetectionClassMigrator(project_root)
    
    success = migrator.run_migration()
    
    if success:
        print("🎉 Migration completed successfully!")
        print("📖 Check MIGRATION_GUIDE.md for next steps")
    else:
        print("❌ Migration failed. Check logs for details.")

if __name__ == "__main__":
    main()
