import unittest
import importlib
import sys
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestImports(unittest.TestCase):
    def setUp(self):
        self.clear_modules()
        logger.info("Starting new test...")
        
    def clear_modules(self):
        for module in list(sys.modules.keys()):
            if module.startswith('MAX'):
                del sys.modules[module]
        logger.info("Cleared all MAX modules from cache")

    def test_independent_imports(self):
        """Test that each module can be imported independently"""
        modules = [
            'MAX.agents.task_expert',
            'MAX.utils.options',
            'MAX.agents.agent',
            'MAX.agents.anthropic_agent',
            'MAX.adapters.llm',
        ]
        
        for module in modules:
            try:
                logger.info(f"Attempting to import {module}")
                importlib.import_module(module)
                logger.info(f"Successfully imported {module}")
                self.assertTrue(True, f"Successfully imported {module}")
            except ModuleNotFoundError as e:
                logger.error(f"Failed to import {module}: {str(e)}")
                self.fail(f"Failed to import {module} independently: {str(e)}")

    def test_import_order_independence(self):
        """Test that modules can be imported in any order"""
        import_orders = [
            ['MAX.utils.options', 'MAX.agents.task_expert'],
            ['MAX.agents.task_expert', 'MAX.utils.options'],
            ['MAX.agents.agent', 'MAX.agents.task_expert', 'MAX.utils.options'],
            ['MAX.utils.options', 'MAX.agents.agent', 'MAX.agents.task_expert']
        ]
        
        for order in import_orders:
            self.clear_modules()
            logger.info(f"Testing import order: {order}")
            try:
                for module in order:
                    importlib.import_module(module)
                logger.info(f"Successfully imported in order: {order}")
                self.assertTrue(True, f"Successfully imported in order: {order}")
            except ImportError as e:
                logger.error(f"Failed to import in order {order}: {str(e)}")
                self.fail(f"Failed to import in order {order}: {str(e)}")

    def test_module_dependencies(self):
        """Test specific module dependencies"""
        logger.info("Testing module dependencies")
        
        # Test task_expert imports
        try:
            from MAX.agents import task_expert
            logger.info("task_expert module imported successfully")
            # Verify it can access its dependencies
            self.assertTrue(hasattr(task_expert, 'TaskExpertAgent'))
            self.assertTrue(hasattr(task_expert, 'TaskExpertOptions'))
        except Exception as e:
            logger.error(f"task_expert dependency test failed: {str(e)}")
            self.fail(f"task_expert dependency test failed: {str(e)}")

        # Test options imports
        try:
            from MAX.utils import options
            logger.info("options module imported successfully")
            # Verify it can access its dependencies
            self.assertTrue(hasattr(options, 'AgentOptions'))
            self.assertTrue(hasattr(options, 'TaskExpertOptions'))
        except Exception as e:
            logger.error(f"options dependency test failed: {str(e)}")
            self.fail(f"options dependency test failed: {str(e)}")

if __name__ == '__main__':
    unittest.main(verbosity=2)