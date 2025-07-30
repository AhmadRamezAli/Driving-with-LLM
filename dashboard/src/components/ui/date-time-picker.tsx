import { useState } from "react";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Calendar as CalendarIcon } from "lucide-react";
import { format } from "date-fns";
import { cn } from "@/lib/utils";

interface DateTimePickerProps {
  value?: Date | string;
  onChange: (date: Date | undefined) => void;
  placeholder?: string;
  disabled?: boolean;
}

export function DateTimePicker({
  value,
  onChange,
  placeholder = "Pick a date",
  disabled = false,
}: DateTimePickerProps) {
  const dateValue = value ? new Date(value) : undefined;
  
  // Handle invalid date
  const isValidDate = dateValue && !isNaN(dateValue.getTime());
  const displayDate = isValidDate ? dateValue : undefined;
  
  const handleTimeChange = (timeString: string) => {
    if (!displayDate && !timeString) return;
    
    const newDate = displayDate ? new Date(displayDate) : new Date();
    if (timeString) {
      const [hours, minutes] = timeString.split(':').map(Number);
      newDate.setHours(hours, minutes);
    }
    onChange(newDate);
  };
  
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className={cn(
            "w-full justify-start text-left font-normal",
            !displayDate && "text-muted-foreground"
          )}
          disabled={disabled}
        >
          <CalendarIcon className="mr-2 h-4 w-4" />
          {displayDate ? (
            format(displayDate, "PPP HH:mm")
          ) : (
            <span>{placeholder}</span>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-0" align="start">
        <Calendar
          mode="single"
          selected={displayDate}
          onSelect={(date) => {
            if (date) {
              // Preserve time from existing date if available
              if (displayDate) {
                date.setHours(
                  displayDate.getHours(),
                  displayDate.getMinutes(),
                  displayDate.getSeconds(),
                  displayDate.getMilliseconds()
                );
              }
              onChange(date);
            } else {
              onChange(undefined);
            }
          }}
          initialFocus
        />
        <div className="p-3 border-t border-border">
          <div className="space-y-2">
            <label className="text-sm font-medium">Time</label>
            <Input
              type="time"
              value={displayDate ? format(displayDate, "HH:mm") : ""}
              onChange={(e) => handleTimeChange(e.target.value)}
            />
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
} 