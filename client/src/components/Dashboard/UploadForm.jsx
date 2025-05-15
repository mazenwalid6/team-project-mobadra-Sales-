import React, { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { useToast } from "../../hooks/use-toast";
import { Plus, X, AlertCircle, Upload, CheckCircle2, Loader2, HelpCircle } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Alert, AlertDescription, AlertTitle } from "../ui/alert";

export default function UploadForm({ onUploadSuccess, onUploadError }) {
  const [file, setFile] = useState(null);
  const [deductions, setDeductions] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [error, setError] = useState(null);
  const [deductionErrors, setDeductionErrors] = useState([]);
  const [showDeductionHelp, setShowDeductionHelp] = useState(false);
  const { toast } = useToast();
  const fileInputRef = React.useRef();

  const validateFile = (file) => {
    if (!file.name.endsWith(".csv")) {
      throw new Error("Please upload a CSV file.");
    }
    if (file.size > 10 * 1024 * 1024) {
      throw new Error("File size must be less than 10MB.");
    }
    return true;
  };

  const validateDeductions = useCallback(() => {
    const errors = deductions.map((deduction, index) => {
      if (!deduction.name.trim()) {
        return "Deduction name is required.";
      }
      if (!deduction.value || isNaN(deduction.value) || Number(deduction.value) < 0) {
        return "Deduction value must be a non-negative number.";
      }
      return null;
    });
    setDeductionErrors(errors);
    return errors.every(error => error === null);
  }, [deductions]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (!file) return;

    try {
      validateFile(file);
      await handleFileUpload(file);
    } catch (err) {
      setError(err.message);
      setUploadStatus("error");
      toast({
        title: "Error",
        description: err.message,
        variant: "destructive",
        duration: 5000,
      });
    }
  }, [toast]);

  const handleFileSelect = useCallback(async (e) => {
    e.preventDefault();
    const file = e.target.files[0];
    if (!file) return;

    try {
      validateFile(file);
      await handleFileUpload(file);
    } catch (err) {
      setError(err.message);
      setUploadStatus("error");
      toast({
        title: "Error",
        description: err.message,
        variant: "destructive",
        duration: 5000,
      });
    }
  }, [toast]);

  const resetAllStates = () => {
    setFile(null);
    setUploadStatus(null);
    setError(null);
    setIsGenerating(false);
    setIsUploading(false);
    setDeductions([]);
    setDeductionErrors([]);
    onUploadSuccess({ isGenerating: false });
  };

  const handleFileUpload = async (file) => {
    if (!file) {
      toast({
        title: "Error",
        description: "Please select a file first",
        variant: "destructive",
        duration: 5000,
      });
      return;
    }

    // Reset all states first
    resetAllStates();
    setIsUploading(true);

    const formData = new FormData();
    formData.append('file', file);
    if (deductions.length > 0) {
      formData.append('deductions', JSON.stringify(deductions));
    }

    try {
      console.log('Starting file upload...');
      const response = await fetch('http://localhost:5001/upload', {
        method: 'POST',
        body: formData,
      });

      console.log('Response received:', response.status);
      const data = await response.json();
      console.log('Response data:', data);

      if (!response.ok) {
        throw new Error(data.error || 'Failed to upload file');
      }

      if (data.status === 'error') {
        throw new Error(data.error || 'Error processing file');
      }

      setFile(file);
      setUploadStatus("success");
      toast({
        title: "Success",
        description: "File uploaded successfully. Click 'Generate Forecast' to proceed.",
        duration: 5000,
      });
    } catch (error) {
      console.error('Upload error:', error);
      setError(error.message);
      toast({
        title: "Error",
        description: error.message || 'Error uploading file',
        variant: "destructive",
        duration: 5000,
      });
    } finally {
      setIsUploading(false);
    }
  };

  const addDeduction = () => {
    setDeductions([...deductions, { name: "", value: "", type: "percentage" }]);
    setDeductionErrors([...deductionErrors, null]);
  };

  const updateDeduction = (index, field, value) => {
    const newDeductions = [...deductions];
    newDeductions[index][field] = value;
    setDeductions(newDeductions);
    validateDeductions();
  };

  const removeDeduction = (index) => {
    setDeductions(deductions.filter((_, i) => i !== index));
    setDeductionErrors(deductionErrors.filter((_, i) => i !== index));
    validateDeductions();
  };

  const handleGenerateForecast = async () => {
    if (!file) return;

    setIsGenerating(true);
    setError(null);
    onUploadSuccess({ isGenerating: true });

    try {
      const formData = new FormData();
      formData.append("file", file);
      if (deductions.length > 0) {
        formData.append("deductions", JSON.stringify(deductions));
      }

      const response = await fetch("http://localhost:5001/detect", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to generate forecast");
      }

      const data = await response.json();
      onUploadSuccess(data);
      setUploadStatus("success");
    } catch (err) {
      console.error("Error generating forecast:", err);
      setError(err.message);
      onUploadError(err.message);
    } finally {
      setIsGenerating(false);
      onUploadSuccess({ isGenerating: false });
    }
  };

  const handleFileButtonClick = (e) => {
    e.preventDefault();
    document.getElementById("file-upload").click();
  };

  return (
    <Card className="border border-border bg-card shadow-none animate-fade-in">
      <CardHeader>
        <CardTitle className="text-base font-semibold text-foreground">Upload Data</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <form onSubmit={(e) => e.preventDefault()} className="space-y-6 p-6">
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>File Requirements</AlertTitle>
            <AlertDescription>
              CSV file with columns: Date (YYYY-MM-DD), Store, Dept, Weekly_Sales, IsHoliday (0/1), Type, Size. Max 10MB.
            </AlertDescription>
          </Alert>

          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {uploadStatus === "success" && (
            <Alert className="mb-4 bg-success/10 border-success/20">
              <CheckCircle2 className="h-4 w-4 text-success" />
              <AlertTitle>File Uploaded Successfully!</AlertTitle>
              <AlertDescription className="text-success">
                Click 'Generate Forecast' to proceed.
              </AlertDescription>
            </Alert>
          )}

          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center ${
              isDragging ? "border-secondary bg-secondary/10" : "border-border hover:border-secondary"
            } transition-colors`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center gap-4">
              <Upload className="h-8 w-8 text-muted-foreground" />
              <div className="space-y-2">
                <p className="text-sm text-foreground">
                  {file ? file.name : "Drag and drop your CSV file here, or click to select"}
                </p>
                <p className="text-xs text-muted-foreground">
                  Supported format: CSV (max 10MB)
                </p>
              </div>
            <Input
              type="file"
                accept=".csv"
                onChange={handleFileSelect}
              className="hidden"
                id="file-upload"
                disabled={isUploading || isGenerating}
            />
                <Button
                  type="button"
                variant="outline"
                onClick={handleFileButtonClick}
                disabled={isUploading || isGenerating}
                className="flex items-center gap-2"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Uploading...
                  </>
                ) : file ? "Change File" : "Select File"}
                </Button>
              </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Label className="text-sm font-medium text-foreground">
              Net Revenue Deductions
            </Label>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => setShowDeductionHelp(!showDeductionHelp)}
                className="h-6 w-6 text-muted-foreground hover:text-foreground"
                aria-label="Toggle deductions help"
              >
                <HelpCircle className="h-4 w-4" />
              </Button>
            </div>
            {showDeductionHelp && (
              <p className="text-xs text-muted-foreground bg-accent/10 p-2 rounded">
                Add deductions like returns or discounts to adjust net revenue calculations.
              </p>
            )}
            <p className="text-xs text-muted-foreground">
              Specify deductions to calculate net revenue (e.g., returns, discounts).
            </p>
            {deductions.length === 0 ? (
              <div className="flex flex-col items-center space-y-2">
                <span className="text-muted-foreground text-sm">No deductions added.</span>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={addDeduction}
                  className="border-border text-foreground hover:bg-accent hover:text-accent-foreground"
                  aria-label="Add deduction"
                >
                  <Plus className="h-4 w-4 mr-2" /> Add Deduction
                </Button>
              </div>
            ) : (
              deductions.map((deduction, index) => (
                <div key={index} className="flex items-start space-x-2 w-full">
                  <div className="flex-1">
                  <Input
                    placeholder="Deduction Name (e.g., Returns)"
                    value={deduction.name}
                    onChange={(e) => updateDeduction(index, "name", e.target.value)}
                      className={`text-sm ${deductionErrors[index] && deduction.name === "" ? "border-destructive" : ""}`}
                    aria-label={`Deduction name ${index + 1}`}
                    maxLength={100}
                  />
                    {deductionErrors[index] && deduction.name === "" && (
                      <p className="text-xs text-destructive mt-1">{deductionErrors[index]}</p>
                    )}
                  </div>
                  <div className="w-20">
                  <Input
                    type="number"
                    placeholder="Value"
                    value={deduction.value}
                    onChange={(e) => updateDeduction(index, "value", e.target.value)}
                      className={`text-sm ${deductionErrors[index] && (!deduction.value || Number(deduction.value) < 0) ? "border-destructive" : ""}`}
                    aria-label={`Deduction value ${index + 1}`}
                      min="0"
                  />
                    {deductionErrors[index] && (!deduction.value || Number(deduction.value) < 0) && (
                      <p className="text-xs text-destructive mt-1">{deductionErrors[index]}</p>
                    )}
                  </div>
                  <Select
                    value={deduction.type}
                    onValueChange={(value) => updateDeduction(index, "type", value)}
                  >
                    <SelectTrigger className="w-24 h-10 text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="percentage">%</SelectItem>
                      <SelectItem value="fixed">$</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => removeDeduction(index)}
                    className="h-10 w-10 text-muted-foreground hover:text-foreground"
                    aria-label={`Remove deduction ${index + 1}`}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ))
            )}
            {deductions.length > 0 && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={addDeduction}
                className="mt-2 border-border text-foreground hover:bg-accent hover:text-accent-foreground"
                aria-label="Add another deduction"
              >
                <Plus className="h-4 w-4 mr-2" /> Add Deduction
              </Button>
            )}
          </div>

          <div className="flex flex-col sm:flex-row gap-2">
            <Button
              type="button"
              onClick={handleGenerateForecast}
              disabled={isGenerating || isUploading || !file || uploadStatus !== "success"}
              className="w-full sm:w-auto flex items-center gap-2"
              aria-label="Generate forecast"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                "Generate Forecast"
              )}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}