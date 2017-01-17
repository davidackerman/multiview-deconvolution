function result = run_tests(varargin)
    test_suite = matlab.unittest.TestSuite.fromFolder('.');
    result = test_suite.run() ;
end
